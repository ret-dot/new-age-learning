#!/usr/bin/env python3
"""
Script to summarize existing video analysis JSONL files.

Usage:
    python summarize_jsonl.py path/to/analysis.jsonl [--output summary.txt]
"""
import json
import jsonlines
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, DefaultDict
from collections import defaultdict, deque
import numpy as np

def load_jsonl_data(file_path: str) -> List[Dict]:
    """Load data from a JSONL file."""
    data = []
    with jsonlines.open(file_path) as reader:
        for obj in reader:
            data.append(obj)
    return data

def detect_scenes(data: List[Dict], min_scene_duration: float = 3.0) -> List[Dict]:
    """Detect distinct scenes based on object changes."""
    if not data:
        return []
        
    scenes = []
    current_scene = {
        'start_time': data[0]['timestamp'],
        'end_time': data[0]['timestamp'],
        'objects': set(),
        'object_changes': []
    }
    
    prev_objects = set()
    
    for frame in data:
        current_objects = {obj['class'] for obj in frame['objects']}
        
        # Check for significant changes
        if frame['timestamp'] - current_scene['end_time'] > min_scene_duration:
            # If scene is long enough, save it
            if current_scene['end_time'] - current_scene['start_time'] >= min_scene_duration:
                scenes.append(current_scene)
            # Start new scene
            current_scene = {
                'start_time': frame['timestamp'],
                'end_time': frame['timestamp'],
                'objects': current_objects,
                'object_changes': []
            }
        
        # Track object changes
        new_objects = current_objects - prev_objects
        removed_objects = prev_objects - current_objects
        
        if new_objects or removed_objects:
            current_scene['object_changes'].append({
                'time': frame['timestamp'],
                'new_objects': list(new_objects),
                'removed_objects': list(removed_objects)
            })
        
        current_scene['end_time'] = frame['timestamp']
        current_scene['objects'].update(current_objects)
        prev_objects = current_objects
    
    # Add the last scene
    if current_scene['end_time'] - current_scene['start_time'] >= min_scene_duration:
        scenes.append(current_scene)
    
    return scenes

def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """Calculate Intersection over Union between two bounding boxes."""
    # box: [x1, y1, x2, y2]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Calculate areas
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Calculate IoU
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def track_object_movements(data: List[Dict], min_movement: float = 0.1) -> Dict[str, List[Dict]]:
    """Track movements of objects across frames, ignoring minor jitters."""
    if not data:
        return {}
    
    # Track objects and their movements
    object_tracks = defaultdict(list)
    object_last_pos = {}  # {object_id: last_bbox}
    
    for frame_idx, frame in enumerate(data):
        current_objects = {}
        
        # First pass: match objects with previous frame
        for obj in frame['objects']:
            obj_class = obj['class']
            bbox = obj['bbox']  # [x1, y1, x2, y2]
            
            # Try to match with previous position
            matched = False
            for obj_id, last_bbox in object_last_pos.items():
                if obj_class == obj_id.split('_')[0]:  # Same class
                    iou = calculate_iou(bbox, last_bbox)
                    if iou > 0.5:  # Consider it the same object if IoU > 0.5
                        object_tracks[obj_id].append({
                            'frame': frame_idx,
                            'time': frame['timestamp'],
                            'bbox': bbox,
                            'movement': 0.0  # Will be updated
                        })
                        current_objects[obj_id] = bbox
                        matched = True
                        break
            
            # If no match, create new track
            if not matched:
                obj_id = f"{obj_class}_{len([k for k in object_last_pos if k.startswith(obj_class)]) + 1}"
                object_tracks[obj_id].append({
                    'frame': frame_idx,
                    'time': frame['timestamp'],
                    'bbox': bbox,
                    'movement': 0.0
                })
                current_objects[obj_id] = bbox
        
        # Update last positions
        object_last_pos = current_objects
    
    # Calculate movement between frames
    for obj_id, track in object_tracks.items():
        if len(track) < 2:
            continue
            
        for i in range(1, len(track)):
            prev_bbox = track[i-1]['bbox']
            curr_bbox = track[i]['bbox']
            
            # Calculate center points
            prev_center = [(prev_bbox[0] + prev_bbox[2])/2, (prev_bbox[1] + prev_bbox[3])/2]
            curr_center = [(curr_bbox[0] + curr_bbox[2])/2, (curr_bbox[1] + curr_bbox[3])/2]
            
            # Calculate Euclidean distance (normalized by frame dimensions)
            movement = np.sqrt((curr_center[0] - prev_center[0])**2 + 
                             (curr_center[1] - prev_center[1])**2)
            
            # Only count significant movements
            if movement > min_movement:
                track[i]['movement'] = movement
    
    return object_tracks

def analyze_object_interactions(data: List[Dict]) -> Tuple[List[tuple], Dict[str, List[Dict]]]:
    """Analyze interactions and movements between objects in the video."""
    interactions = {}
    movement_tracks = track_object_movements(data)
    
    # Analyze co-occurrences
    for frame in data:
        objects = [obj['class'] for obj in frame['objects']]
        for i, obj1 in enumerate(objects):
            for obj2 in objects[i+1:]:
                key = tuple(sorted([obj1, obj2]))
                interactions[key] = interactions.get(key, 0) + 1
    
    return sorted(interactions.items(), key=lambda x: x[1], reverse=True), movement_tracks

def analyze_scene_movements(scene_data: List[Dict], scene_number: int) -> str:
    """Analyze movements within a single scene."""
    if not scene_data:
        return ""
        
    movement_tracks = track_object_movements(scene_data)
    analysis = []
    
    for obj_id, track in movement_tracks.items():
        if len(track) < 2:
            continue
            
        # Calculate total movement
        total_movement = sum(t['movement'] for t in track if 'movement' in t)
        if total_movement < 0.5:  # Skip objects with minimal movement
            continue
            
        # Get movement direction
        first_pos = track[0]['bbox']
        last_pos = track[-1]['bbox']
        dx = last_pos[0] - first_pos[0]  # x movement (normalized)
        dy = last_pos[1] - first_pos[1]  # y movement (normalized)
        
        # Determine direction
        direction = []
        if abs(dx) > 0.1:  # Significant x movement
            direction.append("left" if dx < 0 else "right")
        if abs(dy) > 0.1:  # Significant y movement
            direction.append("up" if dy < 0 else "down")
            
        if not direction:
            direction = ["minimal movement"]
            
        # Calculate speed (movement per second)
        duration = track[-1]['time'] - track[0]['time']
        speed = total_movement / duration if duration > 0 else 0
        
        analysis.append(
            f"- {obj_id.split('_')[0].capitalize()}: Moved {' and '.join(direction)} "
            f"(speed: {speed:.2f} units/sec, distance: {total_movement:.1f} units)"
        )
    
    if not analysis:
        return "\nNo significant movements detected in this scene.\n"
        
    return (
        f"\nScene {scene_number} - Movement Analysis:\n"
        "-" * 40 + "\n"
        + "\n".join(analysis) + "\n"
    )

def generate_narrative(scenes: List[Dict], top_interactions: List[tuple], movement_tracks: Dict[str, List[Dict]]) -> str:
    """Generate a natural language narrative from scene data with motion analysis."""
    if not scenes:
        return "No scenes detected in the video."
    
    narrative = ["Video Narrative Summary", "="*40]
    
    # Add overall description
    total_duration = scenes[-1]['end_time'] - scenes[0]['start_time']
    narrative.append(f"The video is approximately {total_duration:.1f} seconds long and contains {len(scenes)} distinct scenes.")
    
    # Add scene descriptions with movement analysis
    narrative.append("\nScene-by-Scene Analysis:" + "="*60)
    for i, scene in enumerate(scenes, 1):
        duration = scene['end_time'] - scene['start_time']
        objects = ", ".join(sorted(scene['objects']))
        
        # Get scene data for movement analysis
        scene_start = scene['start_time']
        scene_end = scene['end_time']
        scene_frames = [
            frame for frame in data 
            if scene_start <= frame['timestamp'] <= scene_end
        ]
        
        narrative.append(f"\nScene {i} ({duration:.1f}s):" + "-"*50)
        narrative.append(f"• Main objects: {objects}")
        
        # Add movement analysis for this scene
        movement_analysis = analyze_scene_movements(scene_frames, i)
        narrative.append(movement_analysis)
        
        # Describe major changes
        if scene['object_changes']:
            narrative.append("  Key moments:")
            for change in scene['object_changes']:
                time_str = f"{change['time']//60:.0f}:{change['time']%60:02.0f}"
                if change['new_objects']:
                    narrative.append(f"  - At {time_str}: {', '.join(change['new_objects'])} appear")
                if change['removed_objects']:
                    narrative.append(f"  - At {time_str}: {', '.join(change['removed_objects'])} disappear")
    
    # Add motion analysis
    if movement_tracks:
        narrative.append("\nMotion Analysis:" + "-"*35)
        
        # Group tracks by object type
        object_movements = defaultdict(list)
        for obj_id, track in movement_tracks.items():
            obj_class = obj_id.split('_')[0]
            total_movement = sum(t['movement'] for t in track)
            if total_movement > 0.5:  # Only include objects with significant movement
                object_movements[obj_class].append(total_movement)
        
        # Describe movement patterns
        for obj_class, movements in object_movements.items():
            if movements:
                avg_movement = sum(movements) / len(movements)
                narrative.append(f"- {obj_class.capitalize()}(s) showed significant movement "
                              f"(average movement: {avg_movement:.2f} units/frame)")
    
    # Add interaction analysis
    if top_interactions:
        narrative.append("\nKey Object Interactions:" + "-"*35)
        for (obj1, obj2), count in top_interactions[:5]:  # Top 5 interactions
            narrative.append(f"- {obj1.capitalize()} and {obj2} appeared together in {count} frames")
    
    return "\n".join(narrative)

def generate_summary(video_data: List[Dict]) -> str:
    """Generate a comprehensive summary from the analysis data."""
    global data
    data = video_data  # Make data available to other functions
    
    if not data:
        return "No data to summarize."
    
    # Calculate basic statistics
    total_frames = len(data)
    if total_frames <= 1:
        return "Insufficient data for analysis.
    
    duration = data[-1]['timestamp'] - data[0]['timestamp']
    fps = total_frames / duration if duration > 0 else 0
    
    # Detect scenes and analyze interactions/movements
    scenes = detect_scenes(data)
    interactions, movement_tracks = analyze_object_interactions(data)
    
    # Analyze object occurrences
    object_stats = {}
    for frame in data:
        for obj in frame['objects']:
            obj_class = obj['class']
            if obj_class not in object_stats:
                object_stats[obj_class] = {
                    'count': 0,
                    'first_seen': frame['timestamp'],
                    'last_seen': frame['timestamp']
                }
            object_stats[obj_class]['count'] += 1
            object_stats[obj_class]['last_seen'] = frame['timestamp']
    
    # Sort objects by frequency
    sorted_objects = sorted(
        object_stats.items(),
        key=lambda x: x[1]['count'],
        reverse=True
    )
    
    # Generate narrative summary with motion analysis
    narrative = generate_narrative(scenes, interactions, movement_tracks)
    
    # Generate statistical summary
    stats = [
        "Detailed Analysis",
        "=" * 40,
        f"Total frames analyzed: {total_frames}",
        f"Estimated duration: {duration:.1f} seconds",
        f"Average FPS: {fps:.2f}",
        f"Number of scenes detected: {len(scenes)}",
        "\nObject Statistics (Top 10):",
        "-" * 40
    ]
    
    # Add top 10 objects with more details
    for i, (obj, stats) in enumerate(sorted_objects[:10], 1):
        percentage = (stats['count'] / total_frames) * 100
        duration = stats['last_seen'] - stats['first_seen']
        time_first = f"{stats['first_seen']//60:.0f}:{stats['first_seen']%60:02.0f}"
        time_last = f"{stats['last_seen']//60:.0f}:{stats['last_seen']%60:02.0f}"
        
        stats.append(
            f"{i}. {obj.capitalize()}:\n"
            f"   • Appears in {stats['count']} frames ({percentage:.1f}% of video)\n"
            f"   • First seen at {time_first}, last at {time_last}\n"
            f"   • Total screen time: {duration:.1f} seconds"
        )
    
    # Combine narrative and statistics
    summary = [narrative, "\n\n" + "\n".join(stats)]
    
    return "\n".join(summary)

def main():
    parser = argparse.ArgumentParser(description='Summarize video analysis JSONL file.')
    parser.add_argument('input_file', help='Path to the input JSONL file')
    parser.add_argument('--output', '-o', help='Output file (default: print to console)', default=None)
    args = parser.parse_args()
    
    # Load and process data
    print(f"Loading data from {args.input_file}...")
    data = load_jsonl_data(args.input_file)
    print(f"Loaded {len(data)} frames.")
    
    # Generate summary
    print("Generating summary...")
    summary = generate_summary(data)
    
    # Output results
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(summary)
        print(f"Summary saved to {args.output}")
    else:
        print("\n" + "=" * 80)
        print(summary)
        print("=" * 80)

if __name__ == "__main__":
    main()
