import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from skyfield.api import load
import os
import time
from skyfield.api import EarthSatellite

def add_sat_obj_column(df):
    """Add satellite objects column from TLE lines"""
    print("Creating satellite objects from TLE lines...")
    
    # Check if required columns exist
    required_columns = ['tle_line1', 'tle_line2', 'name']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Missing required columns: {missing_columns}")
        print(f"Available columns: {list(df.columns)}")
        return df
    
    df['sat_obj'] = df.apply(
        lambda row: EarthSatellite(row['tle_line1'], row['tle_line2'], row['name']),
        axis=1
    )
    print(f"Successfully created {len(df)} satellite objects")
    return df

class CollisionRiskDashboard:
    def __init__(self):
        self.ts = load.timescale()
        
    def compute_future_positions(self, sat_obj, hours=48, step=1):
        """Compute future positions for a satellite object"""
        times = self.ts.utc(*self.ts.now().utc[:4], range(0, hours, step))
        
        positions = []
        for t in times:
            geocentric = sat_obj.at(t)
            positions.append(geocentric.position.km)
        
        return np.array(positions)
    
    def find_conjunctions(self, debris_df, sat_id, threshold_km=100):
        """Find close approaches between satellites"""
        print(f"Analyzing collision risks for satellite {sat_id}...")
        
        try:
            sat_row = debris_df[debris_df.norad_id == sat_id].iloc[0]
            sat_obj = sat_row['sat_obj']
        except IndexError:
            print(f"Satellite {sat_id} not found in dataset")
            return pd.DataFrame()
        
        sat_positions = self.compute_future_positions(sat_obj, hours=48)
        risks = []
        
        # Check against other debris objects
        for _, debris in debris_df.iterrows():
            if debris['norad_id'] == sat_id:
                continue
            
            try:
                debris_positions = self.compute_future_positions(debris['sat_obj'], hours=48)
                
                for time_idx, (sat_pos, debris_pos) in enumerate(zip(sat_positions, debris_positions)):
                    distance = np.linalg.norm(sat_pos - debris_pos)
                    
                    if distance < threshold_km:
                        # Calculate relative velocity
                        sat_vel = sat_obj.at(self.ts.utc(*self.ts.now().utc[:4], time_idx)).velocity.km_per_s
                        debris_vel = debris['sat_obj'].at(self.ts.utc(*self.ts.now().utc[:4], time_idx)).velocity.km_per_s
                        relative_velocity = np.linalg.norm(sat_vel - debris_vel)
                        
                        risks.append({
                            'time_index': time_idx,
                            'debris_id': debris['norad_id'],
                            'debris_name': debris['name'],
                            'distance_km': distance,
                            'relative_velocity': relative_velocity,
                            'position_x': debris_pos[0],
                            'position_y': debris_pos[1],
                            'position_z': debris_pos[2]
                        })
                        
            except Exception as e:
                print(f"Error processing debris {debris['norad_id']}: {e}")
                continue
        
        return pd.DataFrame(risks)
    
    def create_3d_trajectory_plot(self, sat_id, debris_df, risks_df):
        """Create 3D trajectory plot with collision risks"""
        
        try:
            sat_row = debris_df[debris_df.norad_id == sat_id].iloc[0]
            sat_obj = sat_row['sat_obj']
        except IndexError:
            print(f"Satellite {sat_id} not found")
            return None
        
        # Get satellite trajectory
        sat_trajectory = self.compute_future_positions(sat_obj, hours=48)
        
        # Create 3D plot
        fig = go.Figure()
        
        # Add satellite trajectory
        fig.add_trace(go.Scatter3d(
            x=sat_trajectory[:, 0],
            y=sat_trajectory[:, 1],
            z=sat_trajectory[:, 2],
            mode='lines',
            name=f'{sat_row["name"]} Trajectory',
            line=dict(color='blue', width=3)
        ))
        
        # Add collision risk points
        if not risks_df.empty:
            fig.add_trace(go.Scatter3d(
                x=risks_df['position_x'],
                y=risks_df['position_y'],
                z=risks_df['position_z'],
                mode='markers',
                marker=dict(
                    size=risks_df['distance_km'] / 10,
                    color=risks_df['distance_km'],
                    colorscale='Reds',
                    showscale=True,
                    colorbar=dict(title='Distance (km)')
                ),
                name='Collision Risk',
                text=risks_df['debris_name'],
                hovertemplate='<b>%{text}</b><br>' +
                            'Distance: %{marker.color:.1f} km<br>' +
                            'Relative Velocity: %{customdata:.1f} km/s<br>' +
                            '<extra></extra>',
                customdata=risks_df['relative_velocity']
            ))
        
        fig.update_layout(
            title=f'Collision Risk Assessment for {sat_row["name"]} (48 hours)',
            scene=dict(
                xaxis_title='X (km)',
                yaxis_title='Y (km)',
                zaxis_title='Z (km)',
                aspectmode='data'
            ),
            width=800,
            height=600
        )
        
        return fig
    
    def create_risk_table(self, risks_df):
        """Create risk summary table"""
        if risks_df.empty:
            return "No collision risks detected within the specified threshold."
        
        # Sort by distance
        sorted_risks = risks_df.sort_values('distance_km')
        
        # Create summary table
        summary = sorted_risks[['debris_name', 'distance_km', 'relative_velocity']].head(10)
        
        return summary
    
    def create_simplified_plot(self, sat_id, risks_df):
        """Create simplified 3D plot for faster rendering"""
        # Create a basic plot showing just the risk summary
        fig = go.Figure()
        
        if not risks_df.empty:
            # Create a simple scatter plot of risks
            fig.add_trace(go.Scatter(
                x=list(range(len(risks_df))),
                y=risks_df['min_distance_km'] if 'min_distance_km' in risks_df.columns else [],
                mode='markers+lines',
                name='Collision Risks',
                text=risks_df['debris_name'] if 'debris_name' in risks_df.columns else [],
                hovertemplate='<b>%{text}</b><br>' +
                            'Distance: %{y:.1f} km<br>' +
                            '<extra></extra>'
            ))
            
            fig.update_layout(
                title=f'Top Collision Risks for Satellite {sat_id}',
                xaxis_title='Risk Rank',
                yaxis_title='Minimum Distance (km)',
                width=800,
                height=400
            )
        else:
            # Empty plot
            fig.add_annotation(
                text="No collision risks detected",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False,
                font=dict(size=20)
            )
            fig.update_layout(
                title=f'Collision Risk Assessment for Satellite {sat_id}',
                width=800,
                height=400
            )
        
        return fig
    
    def generate_dashboard(self, sat_id='25544'):  # Default to ISS
        """Generate complete collision risk dashboard using GPU acceleration"""
        import app
        from scripts.gpu_risk_analysis import quick_risk_analysis
        
        try:
            app.add_log(f"Starting GPU-accelerated risk analysis for satellite {sat_id}...", level="info")
            
            # Use the optimized GPU risk analysis
            start_time = time.time()
            result = quick_risk_analysis(str(sat_id), max_objects=3000)
            analysis_time = time.time() - start_time
            
            app.add_log(f"Risk analysis completed in {analysis_time:.2f} seconds", level="success")
            
            risks_df = result['risks_df']
            risk_summary = result['summary']
            
            # Convert summary to proper format for API response
            if isinstance(risk_summary, pd.DataFrame):
                risk_table = risk_summary.to_dict('records')
            elif isinstance(risk_summary, str):
                risk_table = [{'message': risk_summary}]
            else:
                risk_table = []
            
            # Create a simplified 3D plot for top 10 risks only (faster)
            app.add_log("Creating visualization for top risks...", level="info")
            fig = self.create_simplified_plot(str(sat_id), risks_df.head(10) if not risks_df.empty else pd.DataFrame())
            
            app.add_log("Dashboard generation complete.", level="success")
            return {
                'plot': fig,
                'risk_table': risk_table,
                'total_risks': result['total_risks']
            }
            
        except Exception as e:
            app.add_log(f"Error in dashboard generation: {str(e)}", level="error")
            return {
                'plot': None,
                'risk_table': [{'message': f"Error: {str(e)}"}],
                'total_risks': 0
            }

if __name__ == "__main__":
    dashboard = CollisionRiskDashboard()
    
    # Generate dashboard for ISS (25544)
    results = dashboard.generate_dashboard('25544')
    
    if results:
        print(f"Found {results['total_risks']} collision risks")
        print("\nTop 10 collision risks:")
        print(results['risk_table'])
        
        # Save plot
        if results['plot']:
            results['plot'].write_html('data/collision_risk_plot.html')
            print("3D plot saved to data/collision_risk_plot.html")
    else:
        print("Failed to generate dashboard") 