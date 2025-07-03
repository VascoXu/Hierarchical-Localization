"""
Visual Localization Pipeline using HLOC.
Processes videos to extract frames, builds 3D maps, and localizes query images.
"""

import json
import shutil
import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import pycolmap
from hloc import (
    extract_features,
    match_features,
    reconstruction,
    visualization,
    pairs_from_retrieval,
    pairs_from_exhaustive,
)
from hloc.utils import viz_3d
from hloc.localize_sfm import QueryLocalizer, pose_from_cluster

from utils import extract_frames_from_video


class HLoc:
    """Main pipeline for visual localization using HLOC."""
    
    def __init__(self, base_dir: str = "datasets/office"):
        self.base_dir = Path(base_dir)
        self.images_dir = self.base_dir / "mapping"
        self.query_dir = self.base_dir / "query"
        self.outputs_dir = self.base_dir / "outputs"
        
        # configuration
        self.feature_conf = extract_features.confs["disk"]
        self.matcher_conf = match_features.confs["disk+lightglue"]
        self.retrieval_conf = extract_features.confs["netvlad"]
        
        # initialize directories
        self._setup_directories()
    
    def _setup_directories(self):
        """Create necessary directories for the pipeline."""
        self.mapping_dir = self.outputs_dir / "mapping"
        self.query_output_dir = self.outputs_dir / "queries"
        
        self.mapping_dir.mkdir(parents=True, exist_ok=True)
        self.query_output_dir.mkdir(parents=True, exist_ok=True)
    
    def extract_frames(self, video_path: str, output_dir: Path, 
                      frame_rate: int = 1, downsample: int = 2):
        """Extract frames from video for mapping or query."""
        print(f"Extracting frames from {video_path} to {output_dir}")
        extract_frames_from_video(
            video_path=video_path,
            output_dir=output_dir,
            frame_rate=frame_rate,
            downsample=downsample
        )
    
    def get_reference_images(self, exhaustive: bool = True) -> List[str]:
        """Get list of reference images for mapping."""
        if exhaustive:
            # use all available images
            references = [
                p.relative_to(self.images_dir).as_posix() 
                for p in self.images_dir.iterdir() 
                if p.suffix.lower() in ['.jpg', '.jpeg', '.png']
            ]
        else:
            # use NetVLAD to find top-k similar images
            loc_pairs = self.outputs_dir / "pairs-query-netvlad10.txt"
            global_descriptors = extract_features.main(
                self.retrieval_conf, self.images_dir, self.outputs_dir
            )
            pairs_from_retrieval.main(global_descriptors, loc_pairs, num_matched=10)
            references = [
                p.relative_to(self.images_dir).as_posix() 
                for p in self.images_dir.iterdir() 
                if p.suffix.lower() in ['.jpg', '.jpeg', '.png']
            ]
        
        print(f"Found {len(references)} reference images.")
        return references
    
    def build_3d_model(self) -> Tuple[pycolmap.Reconstruction, Path, Path]:
        """Build 3D mapping from reference images."""
        print("Building 3D model...")
        
        sfm_pairs = self.outputs_dir / "pairs-query-netvlad10.txt"
        sfm_dir = self.mapping_dir / "sfm"
        features_path = self.mapping_dir / "features.h5"
        matches_path = self.mapping_dir / "matches.h5"
        
        # check if model already exists
        if self._model_exists(sfm_dir, features_path, matches_path):
            print("✓ Loading existing 3D model...")
            model = pycolmap.Reconstruction(sfm_dir)
            print(f"✓ Loaded model: {model.num_points3D()} points, {model.num_images()} images")
            return model, features_path, matches_path
        
        # extract features and match
        print("Extracting features...")
        extract_features.main(
            self.feature_conf, self.images_dir, self.outputs_dir, 
            feature_path=features_path
        )
        
        print("Matching features...")
        match_features.main(
            self.matcher_conf, sfm_pairs, 
            features=features_path, matches=matches_path
        )
        
        # reconstruct 3D model
        print("Reconstructing 3D model...")
        model = reconstruction.main(
            sfm_dir, self.images_dir, sfm_pairs, features_path, matches_path
        )
        
        print(f"✓ 3D model built: {model.num_points3D()} points, {model.num_images()} images.")
        return model, features_path, matches_path
    
    def _model_exists(self, sfm_dir: Path, features_path: Path, matches_path: Path) -> bool:
        """Check if 3D model files exist."""
        return (
            (sfm_dir / "cameras.bin").exists() and 
            features_path.exists() and 
            matches_path.exists()
        )
    
    def localize_image(self, query_image: str, references: List[str], 
                      model: pycolmap.Reconstruction, 
                      features_mapping: Path, matches_mapping: Path) -> Tuple[Path, Path]:
        """Localize a single query image."""
        print(f"Localizing: {query_image}.")
        
        # Setup paths
        loc_pairs = self.query_output_dir / "pairs-loc.txt"
        features_query = self.query_output_dir / "features.h5"
        matches_query = self.query_output_dir / "matches.h5"
        
        # copy mapping features as base
        shutil.copy2(features_mapping, features_query)
        shutil.copy2(matches_mapping, matches_query)
        
        # extract query features
        extract_features.main(
            self.feature_conf, self.query_dir, 
            image_list=[query_image], feature_path=features_query, overwrite=True
        )
        
        # generate pairs and match
        pairs_from_exhaustive.main(
            loc_pairs, image_list=[query_image], ref_list=references
        )
        match_features.main(
            self.matcher_conf, loc_pairs, 
            features=features_query, matches=matches_query, overwrite=True
        )
        
        return features_query, matches_query
    
    def estimate_pose(self, query_image: str, model: pycolmap.Reconstruction, 
                     references: List[str], features_query: Path, 
                     matches_query: Path) -> Tuple[Optional[Dict], Dict, pycolmap.Camera]:
        """Estimate camera pose for query image."""
        print("Estimating pose...")
        
        # infer camera parameters
        camera = pycolmap.infer_camera_from_image(self.images_dir / query_image)
        
        # get reference image IDs
        ref_ids = [model.find_image_with_name(r).image_id for r in references]
        
        # configure localizer
        conf = {
            "estimation": {"ransac": {"max_error": 12}},
            "refinement": {"refine_focal_length": True, "refine_extra_params": True},
        }
        localizer = QueryLocalizer(model, conf)
        
        # estimate pose
        ret, log = pose_from_cluster(
            localizer, query_image, camera, ref_ids, features_query, matches_query
        )
        
        if ret is not None:
            print(f'✓ Found {ret["num_inliers"]}/{len(ret["inlier_mask"])} inliers.')
        else:
            print("✗ Localization failed!")
        
        return ret, log, camera
    
    def visualize_results(self, model: pycolmap.Reconstruction, 
                         query_results: List[Tuple]) -> Any:
        """Create 3D visualization of all results."""
        print("Creating visualization...")
        
        fig = viz_3d.init_figure()
        viz_3d.plot_reconstruction(
            fig, model, color="rgba(255,0,0,0.5)", 
            name="mapping", points_rgb=True
        )
        
        for query, ret, log, camera in query_results:
            if ret is not None:
                pose = pycolmap.Image(cam_from_world=ret["cam_from_world"])
                viz_3d.plot_camera_colmap(
                    fig, pose, camera, color="rgba(0,255,0,0.5)", 
                    name=query, fill=True, size=5.0
                )
                
                # Plot inlier points
                inlier_points = np.array([
                    model.points3D[pid].xyz 
                    for pid in np.array(log["points3D_ids"])[ret["inlier_mask"]]
                ])
                viz_3d.plot_points(fig, inlier_points, color="lime", ps=1, name=query)
        
        fig.show()
        return fig
    
    def run(self, mapping_video: str, query_video: str, 
                    frame_rate: int = 1, downsample: int = 2, 
                    max_queries: Optional[int] = None) -> List[Tuple]:
        """Run the complete visual localization pipeline."""
        print("Starting Visual Localization Pipeline")
        print("=" * 50)
        
        # Step 1: Extract frames
        print("\n1. Extracting frames from videos...")
        self.extract_frames(mapping_video, self.images_dir, frame_rate, downsample)
        self.extract_frames(query_video, self.query_dir, frame_rate, downsample)
        
        # Step 2: Get reference images
        print("\n2. Getting reference images...")
        references = self.get_reference_images(exhaustive=False)
        
        # Step 3: Build 3D model
        print("\n3. Building 3D model...")
        model, features_mapping, matches_mapping = self.build_3d_model()
        
        # Step 4: Process query images
        print("\n4. Processing query images...")
        query_images = sorted(self.query_dir.glob("frame_*.png"))
        print(f"Found {len(query_images)} query images.")
        
        if max_queries:
            query_images = query_images[:max_queries]
            print(f"Processing first {max_queries} images.")
        
        query_results = []
        for query_path in query_images:
            query_name = query_path.name
            print(f"\nProcessing: {query_name}")
            
            # localize image
            features_query, matches_query = self.localize_image(
                query_name, references, model, features_mapping, matches_mapping
            )
            
            # estimate pose
            ret, log, camera = self.estimate_pose(
                query_name, model, references, features_query, matches_query
            )
            
            # save results
            query_results.append((query_name, ret, log, camera))

      
        # Step 5: Visualize results
        print("\n5. Creating visualization...")
        self.visualize_results(model, query_results)
        
        print("\n" + "=" * 50)
        print("Mapping complete!")
        
        return query_results


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Visual Localization Pipeline using HLOC."
    )
    parser.add_argument(
        "--base_dir", type=str, default="datasets/office",
        help="Base directory for the dataset"
    )
    parser.add_argument(
        "--mapping_video", type=str, default="datasets/office/office.MOV",
        help="Path to mapping video"
    )
    parser.add_argument(
        "--query_video", type=str, default="datasets/office/query/query.MOV",
        help="Path to query video"
    )
    parser.add_argument(
        "--frame_rate", type=int, default=1,
        help="Frame extraction rate"
    )
    parser.add_argument(
        "--downsample", type=int, default=2,
        help="Downsampling factor for frames"
    )
    parser.add_argument(
        "--max_queries", type=int, default=None,
        help="Maximum number of query images to process"
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()
    
    # initialize pipeline
    pipeline = HLoc(args.base_dir)
    
    # run pipeline
    results = pipeline.run(
        mapping_video=args.mapping_video,
        query_video=args.query_video,
        frame_rate=args.frame_rate,
        downsample=args.downsample,
        max_queries=args.max_queries
    )
    
    # print summary
    successful_localizations = sum(1 for _, ret, _, _ in results if ret is not None)
    print(f"\nSummary: {successful_localizations}/{len(results)} successful localizations.")


if __name__ == "__main__":
    main()