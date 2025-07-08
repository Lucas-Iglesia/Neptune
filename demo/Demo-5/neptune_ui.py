"""
Neptune UI Components for the detection system
Provides a modern interface overlay for the OpenCV application
"""

import cv2
import numpy as np
import time
from pathlib import Path
from neptune_config import COLORS, DIMENSIONS, MESSAGES

class NeptuneUI:
    def __init__(self, width=1920, height=1080, logo_path=None):
        self.width = width
        self.height = height
        self.logo_path = logo_path
        self.logo_img = None
        
        # Use colors from configuration
        self.colors = COLORS
        
        # Use dimensions from configuration
        self.header_height = DIMENSIONS['header_height']
        self.stats_panel_width = DIMENSIONS['stats_panel_width']
        self.minimap_width = DIMENSIONS['minimap_width']
        self.minimap_height = DIMENSIONS['minimap_height']
        self.padding = DIMENSIONS['padding']
        
        # Load logo after dimensions are set
        self.load_logo()
        
    def load_logo(self):
        """Load the Neptune logo if path is provided"""
        if self.logo_path and Path(self.logo_path).exists():
            try:
                self.logo_img = cv2.imread(self.logo_path, cv2.IMREAD_UNCHANGED)
                if self.logo_img is not None:
                    # Resize logo to fit in header
                    logo_height = self.header_height - 20
                    aspect_ratio = self.logo_img.shape[1] / self.logo_img.shape[0]
                    logo_width = int(logo_height * aspect_ratio)
                    self.logo_img = cv2.resize(self.logo_img, (logo_width, logo_height))
                    print(f"✅ Logo loaded: {self.logo_path}")
                else:
                    print(f"❌ Failed to load logo: {self.logo_path}")
            except Exception as e:
                print(f"❌ Error loading logo: {e}")
                self.logo_img = None
        else:
            print("ℹ️ No logo path provided or file doesn't exist")
    
    def draw_header(self, frame):
        """Draw the Neptune header with logo and title"""
        # Draw dark header background
        cv2.rectangle(frame, (0, 0), (self.width, self.header_height), self.colors['bg_dark'], -1)
        
        # Draw logo if available
        logo_x = self.padding
        if self.logo_img is not None:
            logo_y = (self.header_height - self.logo_img.shape[0]) // 2
            logo_h, logo_w = self.logo_img.shape[:2]
            
            # Handle transparency if logo has alpha channel
            if self.logo_img.shape[2] == 4:
                # Extract alpha channel and convert to 3-channel mask
                alpha = self.logo_img[:, :, 3] / 255.0
                alpha_3ch = np.stack([alpha, alpha, alpha], axis=2)
                
                # Blend logo with background
                logo_rgb = self.logo_img[:, :, :3]
                bg_region = frame[logo_y:logo_y+logo_h, logo_x:logo_x+logo_w]
                blended = (logo_rgb * alpha_3ch + bg_region * (1 - alpha_3ch)).astype(np.uint8)
                frame[logo_y:logo_y+logo_h, logo_x:logo_x+logo_w] = blended
            else:
                # Direct copy for non-transparent logos
                frame[logo_y:logo_y+logo_h, logo_x:logo_x+logo_w] = self.logo_img
            
            logo_x += logo_w + self.padding
        
        # Draw Neptune title
        title_text = MESSAGES['title']
        title_font = cv2.FONT_HERSHEY_SIMPLEX
        title_scale = 2.0
        title_thickness = 3
        
        (text_width, text_height), baseline = cv2.getTextSize(title_text, title_font, title_scale, title_thickness)
        title_y = (self.header_height + text_height) // 2
        
        cv2.putText(frame, title_text, (logo_x, title_y), title_font, title_scale, 
                   self.colors['primary'], title_thickness)
        
        # Draw header border
        cv2.line(frame, (0, self.header_height-1), (self.width, self.header_height-1), 
                self.colors['border'], 2)
        
        return frame
    
    def draw_stats_panel(self, frame, stats):
        """Draw the statistics panel in the top-left area"""
        panel_x = self.padding
        panel_y = self.header_height + self.padding
        panel_width = self.stats_panel_width
        panel_height = 150
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), 
                     self.colors['bg_light'], -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # Border
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), 
                     self.colors['border'], 2)
        
        # Stats text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_thickness = 2
        line_height = 35
        
        # Extract stats
        active_count = stats.get('active', 0)
        underwater_count = stats.get('underwater', 0)
        danger_count = stats.get('danger', 0)
        max_score = stats.get('max_score', 0)
        
        # Format stats text
        stats_lines = [
            f"Active: {active_count}",
            f"Underwater: {underwater_count}",
            f"Danger: {danger_count}",
            f"Max Score: {max_score}"
        ]
        
        for i, line in enumerate(stats_lines):
            text_y = panel_y + 30 + (i * line_height)
            color = self.colors['text_white']
            
            # Color coding for danger
            if "Danger:" in line and danger_count > 0:
                color = self.colors['danger']
            elif "Underwater:" in line and underwater_count > 0:
                color = self.colors['warning']
            
            cv2.putText(frame, line, (panel_x + 15, text_y), font, font_scale, color, font_thickness)
        
        return frame
    
    def draw_minimap(self, frame, minimap_data, position="top-right"):
        """Draw the minimap with water detection overlay"""
        if position == "top-right":
            map_x = self.width - self.minimap_width - self.padding
            map_y = self.header_height + self.padding
        else:
            map_x = self.padding
            map_y = self.height - self.minimap_height - self.padding
        
        # Draw minimap background
        cv2.rectangle(frame, (map_x, map_y), (map_x + self.minimap_width, map_y + self.minimap_height), 
                     self.colors['bg_light'], -1)
        
        # Draw minimap border
        cv2.rectangle(frame, (map_x, map_y), (map_x + self.minimap_width, map_y + self.minimap_height), 
                     self.colors['border'], 2)
        
        # Draw minimap content if provided
        if minimap_data is not None:
            # Resize minimap to fit
            minimap_resized = cv2.resize(minimap_data, (self.minimap_width, self.minimap_height))
            frame[map_y:map_y + self.minimap_height, map_x:map_x + self.minimap_width] = minimap_resized
        
        return frame
    
    def draw_water_detection_overlay(self, frame, water_mask=None, src_quad=None):
        """Draw water detection visualization overlay"""
        if water_mask is not None:
            # Create colored overlay for water detection
            water_overlay = np.zeros_like(frame)
            water_overlay[water_mask > 0] = self.colors['water_zone']
            
            # Blend with original frame
            cv2.addWeighted(frame, 0.8, water_overlay, 0.2, 0, frame)
        
        if src_quad is not None:
            # Draw water detection quadrilateral
            quad_points = src_quad.astype(np.int32)
            cv2.polylines(frame, [quad_points], True, self.colors['water_zone'], 3)
        
        return frame
    
    def draw_person_detection(self, frame, persons, assignments, tracks):
        """Draw person detection boxes and tracking information"""
        for i, person in enumerate(persons):
            # Get bounding box
            cx, cy, w, h = person.xywh[0].tolist()
            x0, y0 = int(cx - w/2), int(cy - h/2)
            x1, y1 = int(cx + w/2), int(cy + h/2)
            
            # Get track info if assigned
            if i in assignments:
                track_id = assignments[i]
                track = tracks.get(track_id, {})
                
                # Get color based on status and danger level
                status = track.get('status', 'surface')
                dangerosity_score = track.get('dangerosity_score', 0)
                
                if track_id in self.get_danger_tracks(tracks):
                    color = self.colors['danger']
                    thickness = 4
                elif status == 'underwater':
                    color = self.colors['warning']
                    thickness = 3
                else:
                    color = self.colors['success']
                    thickness = 2
                
                # Draw bounding box
                cv2.rectangle(frame, (x0, y0), (x1, y1), color, thickness)
                
                # Draw track info
                status_text = f"ID:{track_id}"
                if status == 'underwater':
                    status_text += " (UNDERWATER)"
                status_text += f" - Score:{dangerosity_score}"
                
                # Add underwater duration if available
                if status == 'underwater' and track.get('underwater_start_time'):
                    duration = time.time() - track['underwater_start_time']
                    status_text += f" | {duration:.1f}s"
                
                # Draw text background
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                font_thickness = 2
                (text_width, text_height), baseline = cv2.getTextSize(status_text, font, font_scale, font_thickness)
                
                cv2.rectangle(frame, (x0, y0 - text_height - 10), (x0 + text_width, y0), 
                             color, -1)
                cv2.putText(frame, status_text, (x0, y0 - 10), font, font_scale, 
                           self.colors['text_white'], font_thickness)
            else:
                # Untracked detection
                cv2.rectangle(frame, (x0, y0), (x1, y1), self.colors['success'], 2)
                conf_text = f"{person.conf.item():.2f}"
                cv2.putText(frame, conf_text, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6, self.colors['text_white'], 2)
        
        return frame
    
    def draw_danger_markers(self, frame, tracks):
        """Draw danger markers for people in critical situations"""
        danger_tracks = self.get_danger_tracks(tracks)
        
        for track_id, track in danger_tracks.items():
            if track.get('dive_point') is not None:
                # Draw red cross at dive point
                dive_x, dive_y = int(track['dive_point'][0]), int(track['dive_point'][1])
                cv2.drawMarker(frame, (dive_x, dive_y), self.colors['danger'], 
                              cv2.MARKER_CROSS, 20, 4)
                
                # Draw pulsing circle for attention
                pulse_radius = int(20 + 10 * abs(np.sin(time.time() * 3)))
                cv2.circle(frame, (dive_x, dive_y), pulse_radius, self.colors['danger'], 2)
        
        return frame
    
    def draw_alert_popup(self, frame, alert_popup):
        """Draw alert popup notifications"""
        alert_popup.update()
        active_alerts = alert_popup.get_active_alerts()
        
        if active_alerts:
            # Position popup under minimap
            popup_x = self.width - self.minimap_width - self.padding
            popup_y = self.header_height + self.minimap_height + self.padding + 20
            popup_width = self.minimap_width
            popup_height = len(active_alerts) * 40 + 20
            
            # Semi-transparent background
            overlay = frame.copy()
            cv2.rectangle(overlay, (popup_x, popup_y), 
                         (popup_x + popup_width, popup_y + popup_height), 
                         self.colors['bg_dark'], -1)
            cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)
            
            # Border
            cv2.rectangle(frame, (popup_x, popup_y), 
                         (popup_x + popup_width, popup_y + popup_height), 
                         self.colors['danger'], 2)
            
            # Alert messages
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 2
            
            for i, alert_msg in enumerate(active_alerts):
                text_y = popup_y + 25 + (i * 40)
                cv2.putText(frame, alert_msg, (popup_x + 10, text_y), 
                           font, font_scale, self.colors['text_white'], font_thickness)
        
        return frame
    
    def draw_danger_scale_legend(self, frame):
        """Draw the danger scale legend"""
        legend_x = self.padding
        legend_y = self.height - 100
        legend_width = 300
        legend_height = 30
        
        # Draw legend background
        cv2.rectangle(frame, (legend_x, legend_y), (legend_x + legend_width, legend_y + legend_height), 
                     self.colors['bg_light'], -1)
        cv2.rectangle(frame, (legend_x, legend_y), (legend_x + legend_width, legend_y + legend_height), 
                     self.colors['border'], 2)
        
        # Draw gradient bar
        for i in range(legend_width):
            score = (i / legend_width) * 100
            color = self.get_color_by_dangerosity(score)
            cv2.line(frame, (legend_x + i, legend_y + 5), (legend_x + i, legend_y + legend_height - 5), 
                    color, 1)
        
        # Draw labels
        cv2.putText(frame, "Safe", (legend_x, legend_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, self.colors['text_white'], 1)
        cv2.putText(frame, "Danger", (legend_x + legend_width - 50, legend_y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text_white'], 1)
        
        return frame
    
    def get_danger_tracks(self, tracks):
        """Get tracks of people in danger"""
        danger_tracks = {}
        current_time = time.time()
        
        for track_id, track in tracks.items():
            if (track.get('status') == 'underwater' and 
                track.get('underwater_start_time') and
                (current_time - track['underwater_start_time']) > 5):  # 5 seconds threshold
                danger_tracks[track_id] = track
        
        return danger_tracks
    
    def get_color_by_dangerosity(self, score):
        """Get color based on dangerosity score"""
        if score <= 20:
            ratio = score / 20.0
            b = int(144 * ratio)
            g = int(100 + (138 * ratio))
            r = int(144 * ratio)
            return (b, g, r)
        elif score <= 40:
            ratio = (score - 20) / 20.0
            b = int(144 * (1 - ratio))
            g = int(238 + (17 * ratio))
            r = int(144 + (111 * ratio))
            return (b, g, r)
        elif score <= 60:
            ratio = (score - 40) / 20.0
            b = 0
            g = int(255 - (90 * ratio))
            r = 255
            return (b, g, r)
        elif score <= 80:
            ratio = (score - 60) / 20.0
            b = 0
            g = int(165 * (1 - ratio))
            r = 255
            return (b, g, r)
        else:
            ratio = (score - 80) / 20.0
            b = 0
            g = 0
            r = int(255 - (116 * ratio))
            return (b, g, r)
    
    def create_frame_with_ui(self, base_frame, stats, minimap_data=None, water_mask=None, 
                           src_quad=None, persons=None, assignments=None, tracks=None, 
                           alert_popup=None):
        """Create a complete frame with Neptune UI overlay"""
        # Create a copy to avoid modifying the original
        frame = base_frame.copy()
        
        # Resize frame to UI dimensions
        frame = cv2.resize(frame, (self.width, self.height))
        
        # Draw UI components
        frame = self.draw_header(frame)
        frame = self.draw_stats_panel(frame, stats)
        frame = self.draw_minimap(frame, minimap_data)
        
        # Draw water detection overlay if enabled
        if water_mask is not None or src_quad is not None:
            frame = self.draw_water_detection_overlay(frame, water_mask, src_quad)
        
        # Draw person detections and tracking
        if persons is not None and assignments is not None and tracks is not None:
            frame = self.draw_person_detection(frame, persons, assignments, tracks)
            frame = self.draw_danger_markers(frame, tracks)
        
        # Draw alert popup
        if alert_popup is not None:
            frame = self.draw_alert_popup(frame, alert_popup)
        
        # Draw danger scale legend
        frame = self.draw_danger_scale_legend(frame)
        
        return frame
