import cv2
import jsonlines
import numpy as np
import argparse
import sys
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import json
from datetime import datetime

@dataclass
class FrameAnalysis:
    frame_number: int
    timestamp: float
    objects: List[Dict[str, Any]]
    scene_description: Optional[str] = None

class VideoAnalyzer:
    def __init__(self, model_name: str = 'yolov8x.pt', target_minutes: float = 30.0):
        """
        Initialize the video analyzer with a YOLO model.
        
        Args:
            model_name: Name of the YOLO model to use (must be compatible with ultralytics)
            target_minutes: Target processing time in minutes (default: 30 minutes)
        """
        try:
            import torch
            from ultralytics import YOLO
            import psutil
            
            # Check for available devices and system resources
            self.device = 'cpu'
            self.cpu_cores = psutil.cpu_count(logical=False) or 1
            self.available_memory = psutil.virtual_memory().available / (1024 ** 3)  # in GB
            
            # Device detection with fallback
            if torch.cuda.is_available():
                self.device = 'cuda:0'
                self.device_name = torch.cuda.get_device_name(0)
                self.half_precision = True
                self.batch_size = 16  # Default for NVIDIA
            elif hasattr(torch, 'is_rocm_available') and torch.is_rocm_available():
                self.device = 'cuda:0'
                self.device_name = 'AMD GPU (ROCm)'
                self.half_precision = True
                self.batch_size = 8  # Default for AMD
            else:
                # CPU-specific optimizations
                self.device_name = 'CPU'
                self.half_precision = False
                self.batch_size = max(1, min(4, self.cpu_cores // 2))  # Use half the physical cores
                # Enable OpenMP threading for CPU
                torch.set_num_threads(max(1, self.cpu_cores // 2))
            
            # Initialize model with optimizations
            self.model = YOLO(model_name).to(self.device)
            self.model.fuse()
            
            # Warmup the model
            warmup_tensor = torch.zeros(1, 3, 640, 640).to(self.device)
            if self.half_precision and self.device != 'cpu':
                warmup_tensor = warmup_tensor.half()
            
            with torch.no_grad():
                self.model(warmup_tensor)
            
            print(f"Using {self.device_name} with batch size {self.batch_size} "
                  f"({self.cpu_cores} CPU cores, {self.available_memory:.1f}GB RAM available)")
            
            # Adaptive processing parameters
            self.target_minutes = target_minutes
            self.last_frame_time = None
            self.avg_frame_time = None
            self.processed_frames = 0
            
        except ImportError as e:
            raise ImportError(f"Required packages not found. Please install with: pip install ultralytics torch psutil")
        
        self.frame_analyses = []
        
    def _process_batch(self, frames_batch, frame_numbers, fps, conf_threshold):
        """Process a batch of frames and return detections with optimizations."""
        if not frames_batch:
            return []
            
        # Convert list of frames to tensor with optimized operations
        batch_tensor = torch.zeros(
            len(frames_batch), 3, 640, 640,  # Fixed size for better optimization
            dtype=torch.float16 if self.half_precision else torch.float32,
            device=self.device
        )
        
        # Preprocess frames in batch
        for i, frame in enumerate(frames_batch):
            # Resize and normalize in one step
            img = cv2.resize(frame, (640, 640))  # Resize to model's expected size
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR to RGB
            img = img.transpose(2, 0, 1)  # HWC to CHW
            img = torch.from_numpy(img).to(self.device)
            
            # Normalize and convert to half precision if needed
            img = img.half() if self.half_precision else img.float()
            batch_tensor[i] = img / 255.0
        
        # Run batch inference with optimized settings
        with torch.no_grad():
            try:
                results = self.model(
                    batch_tensor,
                    conf=conf_threshold,
                    iou=0.45,  # Slightly higher IoU for faster NMS
                    max_det=20,  # Limit max detections per frame
                    verbose=False,
                    agnostic_nms=False,
                    augment=False  # Disable test-time augmentation for speed
                )
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    # Reduce batch size if out of memory
                    print("Out of memory, reducing batch size...")
                    self.batch_size = max(1, self.batch_size // 2)
                    return self._process_batch(frames_batch, frame_numbers, fps, conf_threshold)
                raise
        
        # Process results with optimized loops
        batch_analyses = []
        for i, (frame_num, result) in enumerate(zip(frame_numbers, results)):
            detections = []
            if hasattr(result, 'boxes') and result.boxes is not None:
                boxes = result.boxes
                classes = boxes.cls.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                bboxes = boxes.xyxy.cpu().numpy()
                
                for j in range(len(classes)):
                    detections.append({
                        'class': self.model.names[int(classes[j])],
                        'confidence': float(confs[j]),
                        'bbox': bboxes[j].tolist()
                    })
            
            batch_analyses.append((frame_num, detections))
        
        return batch_analyses

    def _calculate_optimal_frame_interval(self, cap, target_fps: float = 5.0) -> int:
        """Calculate optimal frame interval to meet target FPS."""
        import time
        
        # Sample first few frames to measure processing speed
        sample_frames = 10
        frame_times = []
        
        for _ in range(sample_frames):
            ret, frame = cap.read()
            if not ret:
                break
                
            start_time = time.time()
            # Process a single frame with minimal overhead
            with torch.no_grad():
                self.model(torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255.0)
            frame_times.append(time.time() - start_time)
        
        if not frame_times:
            return 30  # Default fallback
            
        avg_frame_time = sum(frame_times) / len(frame_times)
        target_frame_time = 1.0 / target_fps
        
        # Calculate frame interval needed to achieve target FPS
        frame_interval = max(1, int(avg_frame_time / target_frame_time))
        print(f"Adaptive frame interval set to {frame_interval} (avg frame time: {avg_frame_time*1000:.1f}ms)")
        
        # Reset video capture
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return frame_interval
        
    def _estimate_processing_time(self, total_frames: int, frame_interval: int) -> float:
        """Estimate total processing time in minutes."""
        if not self.avg_frame_time:
            return 0
            
        frames_to_process = total_frames // frame_interval
        return (frames_to_process * self.avg_frame_time) / 60.0
        
    def _adjust_parameters_for_target_time(self, total_frames: int, current_interval: int, elapsed_time: float) -> int:
        """Dynamically adjust frame interval to meet target processing time."""
        if elapsed_time <= 0 or not self.avg_frame_time:
            return current_interval
            
        # Calculate how many frames we can process in remaining time
        remaining_time = (self.target_minutes * 60) - elapsed_time
        if remaining_time <= 0:
            return current_interval * 2  # Double interval if we're over time
            
        processed_fps = self.processed_frames / elapsed_time
        target_fps = (total_frames / (self.target_minutes * 60)) * (current_interval + 1)
        
        # Adjust interval based on current performance
        if processed_fps < target_fps * 0.8:  # If too slow
            return min(current_interval + 1, 60)  # Increase interval, max 60
        elif processed_fps > target_fps * 1.2:  # If too fast
            return max(1, current_interval - 1)  # Decrease interval, min 1
        return current_interval

    def process_video(self, video_path: str, output_dir: str = 'output', 
                     frame_interval: int = None, conf_threshold: float = 0.5) -> str:
        """
        Process a video file and generate analysis with optimizations.
        
        Args:
            video_path: Path to the input video file
            output_dir: Directory to save output files
            frame_interval: Process every N frames (1 = process all frames)
            conf_threshold: Confidence threshold for object detection
            
        Returns:
            Path to the output JSONL file
        """
        import torch
        
        # Create output directory if it doesn't exist
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
                        # Clear batch
                        frames_batch = []
                        frame_numbers = []
                        
                        # Update progress
                        pbar.update(self.batch_size)
                
                frame_count += 1
                
                # Break if we've processed all frames
                if frame_count >= total_frames:
                    break
            
            # Process any remaining frames in the last batch
            if frames_batch:
                batch_analyses = self._process_batch(
                    frames_batch, frame_numbers, fps, conf_threshold
                )
                
                for frame_num, detections in batch_analyses:
                    frame_analysis = FrameAnalysis(
                        frame_number=frame_num,
                        timestamp=frame_num / fps,
                        objects=detections
                    )
                    writer.write(asdict(frame_analysis))
                    self.frame_analyses.append(frame_analysis)
                    processed_count += 1
                
                pbar.update(len(frames_batch))
        
        # Release resources
        cap.release()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        print(f"\nProcessed {processed_count} frames. Results saved to: {output_file}")
        return str(output_file)

    def deduplicate_frames(self, similarity_threshold: float = 0.9) -> List[Dict]:
        """
        Deduplicate similar frames based on object detection results.
        
        Args:
            similarity_threshold: Threshold for considering frames similar (0.0 to 1.0)
            
        Returns:
            List of deduplicated frames with time ranges
        """
        if not self.frame_analyses:
            raise ValueError("No frame analyses available. Process a video first.")
            
        # Simple deduplication based on object counts and types
        # More advanced version could use feature hashing or embeddings
        unique_frames = []
        current_group = [self.frame_analyses[0]]
        
        for i in range(1, len(self.frame_analyses)):
            current_frame = self.frame_analyses[i]
            last_unique_frame = current_group[-1]
            
            # Simple similarity check: same number of objects with same classes
            current_objects = {obj['class'] for obj in current_frame.objects}
            last_objects = {obj['class'] for obj in last_unique_frame.objects}
            
            # Calculate Jaccard similarity
            if current_objects and last_objects:
                similarity = len(current_objects.intersection(last_objects)) / len(current_objects.union(last_objects))
            else:
                similarity = 1.0 if not current_objects and not last_objects else 0.0
                
            if similarity >= similarity_threshold:
                current_group.append(current_frame)
            else:
                # Add the group to unique frames with time range
                unique_frames.append({
                    'start_frame': current_group[0].frame_number,
                    'end_frame': current_group[-1].frame_number,
                    'start_time': current_group[0].timestamp,
                    'end_time': current_group[-1].timestamp,
                    'duration': current_group[-1].timestamp - current_group[0].timestamp,
                    'objects': [obj['class'] for obj in current_group[0].objects],
                    'object_count': len(current_group[0].objects)
                })
                current_group = [current_frame]
                
        # Add the last group
        if current_group:
            unique_frames.append({
                'start_frame': current_group[0].frame_number,
                'end_frame': current_group[-1].frame_number,
                'start_time': current_group[0].timestamp,
                'end_time': current_group[-1].timestamp,
                'duration': current_group[-1].timestamp - current_group[0].timestamp,
                'objects': [obj['class'] for obj in current_group[0].objects],
                'object_count': len(current_group[0].objects)
            })
            
        return unique_frames
    
    def generate_summary(self, deduplicated_frames: List[Dict], model_name: str = "gpt-4") -> str:
        """
        Generate a summary of the video using either an LLM or logical analysis.
        
        Args:
            deduplicated_frames: List of deduplicated frames
            model_name: Name of the LLM model to use (if available)
            
        Returns:
            Generated summary text
        """
        # First generate the logical explanation
        logical_explanation = self._generate_logical_explanation(deduplicated_frames)
        
        # If OpenAI is available, enhance with LLM
        try:
            from openai import OpenAI
            client = OpenAI()
            
            # Prepare the prompt with the logical analysis
            prompt = """You are a video analysis assistant. Below is a logical analysis of the video content:
            
            {}
            
            Please provide a more natural, human-readable summary based on this analysis. 
            Focus on creating a coherent narrative of what's happening in the video.
            """.format(logical_explanation)
            
            # Call the OpenAI API
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that summarizes video content in a natural, engaging way."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Using logical analysis (LLM not available: {str(e)})")
            return logical_explanation
    
    def _generate_logical_explanation(self, deduplicated_frames: List[Dict]) -> str:
        """Generate a logical explanation of the video content based on object detections."""
        if not deduplicated_frames:
            return "No significant content was detected in the video."
            
        # Analyze object occurrences
        object_stats = {}
        for segment in deduplicated_frames:
            for obj in segment['objects']:
                if obj not in object_stats:
                    object_stats[obj] = {
                        'count': 0,
                        'total_duration': 0,
                        'first_seen': segment['start_time'],
                        'last_seen': segment['start_time'] + segment['duration']
                    }
                object_stats[obj]['count'] += 1
                object_stats[obj]['total_duration'] += segment['duration']
                object_stats[obj]['last_seen'] = segment['start_time'] + segment['duration']
        
        # Sort objects by total duration
        sorted_objects = sorted(
            object_stats.items(),
            key=lambda x: x[1]['total_duration'],
            reverse=True
        )
        
        # Generate explanation
        explanation = []
        explanation.append("Video Content Analysis:")
        explanation.append("=" * 40)
        
        # Main objects analysis
        if sorted_objects:
            main_objects = [obj for obj, _ in sorted_objects[:3] if object_stats[obj]['total_duration'] > 10]
            if main_objects:
                explanation.append(f"\nMain Focus: The video primarily features {', '.join(main_objects[:-1])}{' and ' if len(main_objects) > 1 else ''}{main_objects[-1]}.")
        
        # Duration analysis
        total_video_duration = max(seg['end_time'] for seg in deduplicated_frames)
        explanation.append(f"\nDuration Analysis (Total: {total_video_duration/60:.1f} minutes):")
        
        for obj, stats in sorted_objects[:5]:  # Top 5 objects
            percentage = (stats['total_duration'] / total_video_duration) * 100
            explanation.append(
                f"- {obj}: Appeared {stats['count']} times, "
                f"visible for {stats['total_duration']/60:.1f} minutes ({percentage:.1f}% of video)"
            )
        
        # Scene changes
        scene_changes = []
        prev_objects = set()
        
        for i, segment in enumerate(deduplicated_frames):
            current_objects = set(segment['objects'])
            if i > 0 and current_objects != prev_objects:
                time_str = f"{segment['start_time']//60:.0f}:{segment['start_time']%60:02.0f}"
                new_objects = current_objects - prev_objects
                removed_objects = prev_objects - current_objects
                
                change = [f"At {time_str}:"]
                if new_objects:
                    change.append(f"{', '.join(new_objects)} appeared")
                if removed_objects:
                    if new_objects:
                        change.append("and")
                    change.append(f"{', '.join(removed_objects)} disappeared")
                
                scene_changes.append(" ".join(change))
            prev_objects = current_objects
        
        if scene_changes:
            explanation.append("\nKey Scene Changes:")
            for i, change in enumerate(scene_changes[:5], 1):  # Show top 5 changes
                explanation.append(f"{i}. {change}")
        
        return "\n".join(explanation)
        

def main():
    import argparse
    import sys
    from pathlib import Path
    import json

    parser = argparse.ArgumentParser(description='Analyze and summarize videos')
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze a video file')
    analyze_parser.add_argument('video_path', help='Path to input video file')
    analyze_parser.add_argument('--output-dir', default='output', help='Directory to save output files')
    analyze_parser.add_argument('--frame-interval', type=int, default=30, 
                              help='Process every N frames (default: 30)')
    analyze_parser.add_argument('--conf-threshold', type=float, default=0.5,
                              help='Confidence threshold for object detection (default: 0.5)')
    analyze_parser.add_argument('--similarity-threshold', type=float, default=0.8,
                              help='Threshold for considering frames similar (0.0 to 1.0)')
    
    args = parser.parse_args()
    
    if args.command == 'analyze':
        # Initialize the analyzer
        analyzer = VideoAnalyzer()
        
        # Process the video
        print(f"Processing video: {args.video_path}")
        output_file = analyzer.process_video(
            args.video_path,
            output_dir=args.output_dir,
            frame_interval=args.frame_interval,
            conf_threshold=args.conf_threshold
        )
        
        if output_file:
            print(f"\nAnalysis complete! Results saved to: {output_file}")
            
            # Deduplicate frames
            deduplicated = analyzer.deduplicate_frames(similarity_threshold=args.similarity_threshold)
            
            # Generate and print summary
            summary = analyzer.generate_summary(deduplicated)
            print("\n=== Video Summary ===")
            print(summary)
            
            # Save deduplicated results
            dedup_file = Path(output_file).with_stem(Path(output_file).stem + "_deduplicated")
            with open(dedup_file, 'w') as f:
                json.dump(deduplicated, f, indent=2)
            print(f"\nDeduplicated results saved to: {dedup_file}")
        else:
            print("Failed to process video.")
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
