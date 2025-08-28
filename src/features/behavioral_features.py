import pandas as pd
import numpy as np
from typing import Dict
import logging

logger = logging.getLogger(__name__)

class BehavioralFeatureEngine:
    """Advanced behavioral pattern features"""
    
    def __init__(self):
        pass
    
    def create_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create behavioral pattern features"""
        logger.info("Creating behavioral features...")
        
        behavioral_features = []
        
        for session_id, group in df.groupby('user_session'):
            group = group.sort_values('event_time')
            features = self._extract_behavioral_features(group, session_id)
            behavioral_features.append(features)
        
        result_df = pd.DataFrame(behavioral_features)
        logger.info(f"Created {len(result_df.columns)-1} behavioral features")
        
        return result_df
    
    def _extract_behavioral_features(self, group: pd.DataFrame, session_id: str) -> Dict:
        """Extract behavioral features for a single session"""
        
        event_seq = group['event_type'].tolist()
        events_str = '->'.join(event_seq)
        product_seq = group['product_id'].tolist()
        
        # High-value behavioral patterns
        has_multiple_buys = int(events_str.count('BUY') > 1)
        has_buy_end = int(events_str.endswith('BUY'))
        has_addcart_buy_pattern = int('ADD_CART->BUY' in events_str)
        has_view_buy_pattern = int('VIEW->BUY' in events_str)
        
        # Decision patterns
        indecision_count = events_str.count('ADD_CART->REMOVE_CART')
        
        # Product revisit behavior
        product_revisits = 0
        seen_products = set()
        for product in product_seq:
            if product in seen_products:
                product_revisits += 1
            seen_products.add(product)
        
        # Category exploration
        category_changes = 0
        prev_category = None
        for category in group['category_id']:
            if prev_category is not None and category != prev_category:
                category_changes += 1
            prev_category = category
        
        # Timing behavior
        if len(group) > 1:
            time_diffs = group['event_time'].diff().dt.total_seconds().dropna()
            max_time_gap = time_diffs.max()
            has_long_pause = int((time_diffs > 300).any())  # 5+ minute pause
            has_quick_actions = int((time_diffs < 10).any())  # Under 10 seconds
        else:
            max_time_gap = 0
            has_long_pause = 0
            has_quick_actions = 0
        
        # Shopping archetypes
        unique_products = group['product_id'].nunique()
        unique_categories = group['category_id'].nunique()
        
        is_focused_shopper = int(unique_products == 1 and len(event_seq) > 1)
        is_comparison_shopper = int(unique_products >= 3 and unique_categories <= 2)
        is_explorer = int(unique_categories >= 3)
        
        # Conversion efficiency
        buy_events = event_seq.count('BUY')
        cart_events = event_seq.count('ADD_CART')
        
        if cart_events > 0:
            cart_conversion_rate = buy_events / cart_events
        else:
            cart_conversion_rate = 0
        
        return {
            'user_session': session_id,
            
            # High-value patterns
            'has_multiple_buys': has_multiple_buys,
            'has_buy_end': has_buy_end,
            'has_addcart_buy_pattern': has_addcart_buy_pattern,
            'has_view_buy_pattern': has_view_buy_pattern,
            
            # Decision patterns
            'indecision_count': indecision_count,
            'product_revisits': product_revisits,
            'category_changes': category_changes,
            
            # Timing patterns
            'max_time_gap': max_time_gap,
            'has_long_pause': has_long_pause,
            'has_quick_actions': has_quick_actions,
            
            # Shopping types
            'is_focused_shopper': is_focused_shopper,
            'is_comparison_shopper': is_comparison_shopper,
            'is_explorer': is_explorer,
            
            # Efficiency metrics
            'cart_conversion_rate': cart_conversion_rate,
            'revisit_ratio': product_revisits / max(unique_products, 1)
        }