import pandas as pd
import numpy as np
from typing import Dict
import logging

logger = logging.getLogger(__name__)

class TemporalFeatureEngine:
    """Time-based features for session analysis"""
    
    def __init__(self):
        pass
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create temporal-based features"""
        logger.info("Creating temporal features...")
        
        temporal_features = []
        
        for session_id, group in df.groupby('user_session'):
            group = group.sort_values('event_time')
            features = self._extract_temporal_features(group, session_id)
            temporal_features.append(features)
        
        result_df = pd.DataFrame(temporal_features)
        logger.info(f"Created {len(result_df.columns)-1} temporal features")
        
        return result_df
    
    def _extract_temporal_features(self, group: pd.DataFrame, session_id: str) -> Dict:
        """Extract temporal features for a single session"""
        
        first_time = group.iloc[0]['event_time']
        
        # Basic time features
        hour = first_time.hour
        day_of_week = first_time.dayofweek  # 0=Monday
        day = first_time.day
        month = first_time.month
        
        # Time period indicators
        is_weekend = int(day_of_week >= 5)
        is_monday = int(day_of_week == 0)
        is_friday = int(day_of_week == 4)
        is_sunday = int(day_of_week == 6)
        
        # Hour-based features
        is_night = int(hour >= 22 or hour <= 6)
        is_morning = int(6 < hour <= 12)
        is_afternoon = int(12 < hour <= 18)
        is_evening = int(18 < hour <= 22)
        is_business_hours = int(9 <= hour <= 17)
        is_lunch_time = int(11 <= hour <= 14)
        
        # Research-based high-value time periods
        is_peak_hour = int(hour in [10, 11, 12, 15])  # High-value hours from analysis
        is_prime_shopping = int(19 <= hour <= 22)  # Evening shopping
        
        # Month-based patterns
        is_month_start = int(day <= 5)
        is_month_end = int(day >= 25)
        is_mid_month = int(13 <= day <= 17)  # Payday periods
        is_payday = int(day in [15, 30, 31])
        
        # Seasonal features
        is_summer = int(month in [6, 7, 8])
        is_winter = int(month in [12, 1, 2])
        
        # Session timing patterns
        if len(group) > 1:
            session_span_hours = (group['event_time'].max() - group['event_time'].min()).total_seconds() / 3600
            crosses_hour_boundary = int(session_span_hours >= 1)
            crosses_day_boundary = int(first_time.date() != group.iloc[-1]['event_time'].date())
        else:
            session_span_hours = 0
            crosses_hour_boundary = 0
            crosses_day_boundary = 0
        
        # Time-based shopping behavior
        is_quick_session = int(session_span_hours < 0.1)  # Under 6 minutes
        is_extended_session = int(session_span_hours > 2)  # Over 2 hours
        
        return {
            'user_session': session_id,
            
            # Basic time
            'hour': hour,
            'day_of_week': day_of_week,
            'day': day,
            'month': month,
            
            # Day type
            'is_weekend': is_weekend,
            'is_monday': is_monday,
            'is_friday': is_friday,
            'is_sunday': is_sunday,
            
            # Hour type
            'is_night': is_night,
            'is_morning': is_morning,
            'is_afternoon': is_afternoon,
            'is_evening': is_evening,
            'is_business_hours': is_business_hours,
            'is_lunch_time': is_lunch_time,
            
            # High-value periods
            'is_peak_hour': is_peak_hour,
            'is_prime_shopping': is_prime_shopping,
            
            # Monthly patterns
            'is_month_start': is_month_start,
            'is_month_end': is_month_end,
            'is_mid_month': is_mid_month,
            'is_payday': is_payday,
            
            # Seasonal
            'is_summer': is_summer,
            'is_winter': is_winter,
            
            # Session timing
            'session_span_hours': session_span_hours,
            'crosses_hour_boundary': crosses_hour_boundary,
            'crosses_day_boundary': crosses_day_boundary,
            'is_quick_session': is_quick_session,
            'is_extended_session': is_extended_session
        }