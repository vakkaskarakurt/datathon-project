import pandas as pd
import numpy as np
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class BaseFeatureEngine:
    """Basic session-level features with strong predictive power"""
    
    def __init__(self):
        pass
    
    def create_session_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create basic session-level features"""
        logger.info("Creating base session features...")
        
        session_features = []
        
        for session_id, group in df.groupby('user_session'):
            group = group.sort_values('event_time')
            features = self._extract_basic_features(group, session_id)
            session_features.append(features)
        
        result_df = pd.DataFrame(session_features)
        logger.info(f"Created {len(result_df.columns)-2} base features for {len(result_df)} sessions")
        
        return result_df
    
    def _extract_basic_features(self, group: pd.DataFrame, session_id: str) -> Dict:
        """Extract basic features for a single session"""
        
        # Basic counts
        total_events = len(group)
        unique_products = group['product_id'].nunique()
        unique_categories = group['category_id'].nunique()
        
        # Event type counts
        event_counts = group['event_type'].value_counts()
        buy_count = event_counts.get('BUY', 0)
        addcart_count = event_counts.get('ADD_CART', 0)
        view_count = event_counts.get('VIEW', 0)
        remove_count = event_counts.get('REMOVE_CART', 0)
        
        # Ratios
        buy_ratio = buy_count / total_events if total_events > 0 else 0
        addcart_ratio = addcart_count / total_events if total_events > 0 else 0
        view_ratio = view_count / total_events if total_events > 0 else 0
        remove_ratio = remove_count / total_events if total_events > 0 else 0
        
        # First and last events
        first_event = group.iloc[0]['event_type']
        last_event = group.iloc[-1]['event_type']
        
        # Session duration
        if len(group) > 1:
            duration_minutes = (group['event_time'].max() - group['event_time'].min()).total_seconds() / 60
        else:
            duration_minutes = 0
        
        # Event velocity
        events_per_minute = total_events / max(duration_minutes, 0.01)
        
        return {
            'user_session': session_id,
            'user_id': group.iloc[0]['user_id'],
            
            # Basic counts
            'total_events': total_events,
            'unique_products': unique_products,
            'unique_categories': unique_categories,
            
            # Event type counts
            'buy_count': buy_count,
            'addcart_count': addcart_count,
            'view_count': view_count,
            'remove_count': remove_count,
            
            # Ratios
            'buy_ratio': buy_ratio,
            'addcart_ratio': addcart_ratio,
            'view_ratio': view_ratio,
            'remove_ratio': remove_ratio,
            
            # Event indicators
            'last_event_buy': int(last_event == 'BUY'),
            'last_event_addcart': int(last_event == 'ADD_CART'),
            'last_event_view': int(last_event == 'VIEW'),
            'last_event_remove': int(last_event == 'REMOVE_CART'),
            'first_event_view': int(first_event == 'VIEW'),
            'first_event_addcart': int(first_event == 'ADD_CART'),
            
            # Temporal
            'duration_minutes': duration_minutes,
            'events_per_minute': events_per_minute,
            
            # Target
            'session_value': group.iloc[0].get('session_value', np.nan)
        }