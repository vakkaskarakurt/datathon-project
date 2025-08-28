import pandas as pd
import numpy as np
from typing import Dict, Optional

class ResearchBasedFeatureEngine:
    """
    Feature engineering based on e-commerce behavior research
    and session value prediction literature
    """
    
    def __init__(self):
        self.category_page_values: Dict[str, float] = {}
    
    def fit_category_values(self, df: pd.DataFrame) -> 'ResearchBasedFeatureEngine':
        """Learn category value patterns from training data"""
        category_session_values = df.groupby(['user_session', 'category_id']).agg({
            'session_value': 'first'
        }).groupby('category_id')['session_value'].mean().to_dict()
        
        self.category_page_values = category_session_values
        return self
    
    def create_session_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create research-based features for session value prediction"""
        
        session_features = []
        
        for session_id, group in df.groupby('user_session'):
            group = group.sort_values('event_time')
            features = self._extract_session_features(group, session_id)
            session_features.append(features)
        
        return pd.DataFrame(session_features)
    
    def _extract_session_features(self, group: pd.DataFrame, session_id: str) -> Dict:
        """Extract all research-based features for a single session"""
        
        # Duration and intensity features (critical predictors)
        duration_features = self._get_duration_features(group)
        
        # Page value simulation (from web analytics literature)
        page_value_features = self._get_page_value_features(group)
        
        # Bounce rate and exit indicators
        bounce_features = self._get_bounce_features(group)
        
        # Sequential behavior patterns
        sequence_features = self._get_sequence_features(group)
        
        # RFM-inspired features
        rfm_features = self._get_rfm_features(group)
        
        # Temporal behavior patterns
        temporal_features = self._get_temporal_features(group)
        
        # Product engagement depth
        engagement_features = self._get_engagement_features(group)
        
        # Decision-making indicators
        decision_features = self._get_decision_features(group)
        
        # Exploration vs exploitation patterns
        exploration_features = self._get_exploration_features(group)
        
        # Combine all features
        return {
            'user_session': session_id,
            **duration_features,
            **page_value_features,
            **bounce_features,
            **sequence_features,
            **rfm_features,
            **temporal_features,
            **engagement_features,
            **decision_features,
            **exploration_features,
            'session_value': group.iloc[0].get('session_value', np.nan)
        }
    
    def _get_duration_features(self, group: pd.DataFrame) -> Dict:
        """Session duration and event timing features"""
        if len(group) > 1:
            duration_seconds = (group['event_time'].max() - group['event_time'].min()).total_seconds()
            duration_minutes = duration_seconds / 60
            time_diffs = group['event_time'].diff().dt.total_seconds().dropna()
        else:
            duration_seconds = duration_minutes = 0
            time_diffs = pd.Series([0])
        
        return {
            'session_duration_minutes': duration_minutes,
            'session_duration_seconds': duration_seconds,
            'avg_time_between_events': time_diffs.mean(),
            'std_time_between_events': time_diffs.std() if len(time_diffs) > 1 else 0,
            'max_time_gap': time_diffs.max(),
            'event_intensity': len(group) / max(duration_minutes, 0.01)
        }
    
    def _get_page_value_features(self, group: pd.DataFrame) -> Dict:
        """Simulate page value based on category performance"""
        categories = group['category_id'].unique()
        
        if len(categories) > 0:
            page_values = [self.category_page_values.get(cat, 0) for cat in categories]
            return {
                'total_page_value': sum(page_values),
                'max_page_value': max(page_values),
                'avg_page_value': np.mean(page_values)
            }
        
        return {'total_page_value': 0, 'max_page_value': 0, 'avg_page_value': 0}
    
    def _get_sequence_features(self, group: pd.DataFrame) -> Dict:
        """Sequential behavior pattern features"""
        events_sequence = group['event_type'].tolist()
        events_str = '->'.join(events_sequence)
        
        return {
            'has_view_addcart_buy_sequence': int('VIEW->ADD_CART->BUY' in events_str),
            'has_multiple_addcart_sequence': int(events_str.count('ADD_CART') >= 2),
            'has_comparison_pattern': int(len(set(group['category_id'])) > 2),
            'funnel_addcart_to_buy': events_str.count('ADD_CART->BUY') / max(events_str.count('ADD_CART'), 1)
        }
    
    # ... additional feature extraction methods