import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class ResearchBasedFeatureEngine:
    """Research-based features with highest correlation to session value"""
    
    def __init__(self):
        self.category_page_values: Dict[str, float] = {}
        self.product_page_values: Dict[str, float] = {}
        
    def fit_category_values(self, df: pd.DataFrame) -> 'ResearchBasedFeatureEngine':
        """Learn category and product value patterns from training data"""
        logger.info("Learning category and product value patterns...")
        
        # Category page values
        category_session_values = df.groupby(['user_session', 'category_id']).agg({
            'session_value': 'first'
        }).groupby('category_id')['session_value'].mean().to_dict()
        
        # Product page values  
        product_session_values = df.groupby(['user_session', 'product_id']).agg({
            'session_value': 'first'  
        }).groupby('product_id')['session_value'].mean().to_dict()
        
        self.category_page_values = category_session_values
        self.product_page_values = product_session_values
        
        logger.info(f"Learned values for {len(self.category_page_values)} categories and {len(self.product_page_values)} products")
        
        return self
    
    def create_research_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create research-based features with highest predictive power"""
        logger.info("Creating research-based features...")
        
        session_features = []
        
        for session_id, group in df.groupby('user_session'):
            group = group.sort_values('event_time')
            features = self._extract_research_features(group, session_id)
            session_features.append(features)
        
        result_df = pd.DataFrame(session_features)
        logger.info(f"Created {len(result_df.columns)-2} research-based features")
        
        return result_df
    
    def _extract_research_features(self, group: pd.DataFrame, session_id: str) -> Dict:
        """Extract all research-based features for a single session"""
        
        # Basic session info
        events_sequence = group['event_type'].tolist()
        events_str = '->'.join(events_sequence)
        
        # 1. MONETARY PROXY (Correlation: 0.792)
        monetary_proxy = events_str.count('BUY')
        
        # 2. TOTAL PAGE VALUE (Correlation: 0.524) 
        categories = group['category_id'].unique()
        products = group['product_id'].unique()
        
        if len(categories) > 0:
            category_values = [self.category_page_values.get(cat, 0) for cat in categories]
            total_page_value = sum(category_values)
            max_category_value = max(category_values)
            avg_category_value = np.mean(category_values)
        else:
            total_page_value = max_category_value = avg_category_value = 0
            
        if len(products) > 0:
            product_values = [self.product_page_values.get(prod, 0) for prod in products]
            max_product_value = max(product_values)
            avg_product_value = np.mean(product_values)
        else:
            max_product_value = avg_product_value = 0
        
        # 3. DECISIVENESS SCORE (Correlation: 0.473)
        if 'BUY' in events_sequence:
            first_buy_position = events_sequence.index('BUY')
            # Earlier purchase = more decisive (higher score)
            decisiveness_score = 1.0 - (first_buy_position / len(events_sequence))
        else:
            decisiveness_score = 0.0
        
        # 4. COMPARISON PATTERN (Correlation: 0.411)
        has_comparison_pattern = int(len(set(group['category_id'])) > 2)
        
        # 5. BOUNCE INDICATORS (Correlation: 0.311)
        is_bounce = int(len(group) == 1)
        has_immediate_exit = int(len(group) == 1 and group.iloc[0]['event_type'] == 'VIEW')
        
        # 6. PAGE VALUE VARIANCE (Research shows high-variance sessions = high value)
        if len(category_values) > 1:
            std_category_value = np.std(category_values)
        else:
            std_category_value = 0
            
        # 7. SESSION INTENSITY FEATURES
        if len(group) > 1:
            session_duration_minutes = (group['event_time'].max() - group['event_time'].min()).total_seconds() / 60
            time_diffs = group['event_time'].diff().dt.total_seconds().dropna()
            
            max_time_gap = time_diffs.max()
            std_time_between_events = time_diffs.std() if len(time_diffs) > 1 else 0
            avg_time_between_events = time_diffs.mean()
        else:
            session_duration_minutes = 0
            max_time_gap = 0
            std_time_between_events = 0
            avg_time_between_events = 0
        
        # 8. PRODUCT ENGAGEMENT DEPTH
        products_viewed = group[group['event_type'] == 'VIEW']['product_id'].nunique()
        if products_viewed > 0 and session_duration_minutes > 0:
            time_per_product = session_duration_minutes / products_viewed
        else:
            time_per_product = 0
            
        # Product interaction intensity
        product_interactions = group['product_id'].value_counts()
        max_product_interactions = product_interactions.max() if len(product_interactions) > 0 else 0
        
        # 9. EXPLORATION PATTERNS
        unique_categories = group['category_id'].nunique()
        unique_products = group['product_id'].nunique()
        
        exploration_score = (unique_categories + unique_products) / len(group)
        category_diversity = unique_categories / max(unique_products, 1)
        
        # 10. ADVANCED BEHAVIORAL PATTERNS
        decision_changes = events_str.count('ADD_CART->REMOVE_CART')
        
        # Multiple add-to-cart sequence (indicates serious consideration)
        has_multiple_addcart_sequence = int(events_str.count('ADD_CART') >= 2)
        
        # Full funnel completion
        has_view_addcart_buy_sequence = int('VIEW->ADD_CART->BUY' in events_str)
        
        # Conversion funnel efficiency
        funnel_addcart_to_buy = events_str.count('ADD_CART->BUY') / max(events_str.count('ADD_CART'), 1)
        
        # 11. HIGH-VALUE CATEGORY INDICATORS
        has_high_value_category = int(max_category_value > np.percentile(list(self.category_page_values.values()), 75))
        has_ultra_high_value_category = int(max_category_value > np.percentile(list(self.category_page_values.values()), 90))
        
        return {
            'user_session': session_id,
            
            # Top predictive features from research
            'monetary_proxy': monetary_proxy,
            'total_page_value': total_page_value,
            'decisiveness_score': decisiveness_score,
            'has_comparison_pattern': has_comparison_pattern,
            
            # Bounce and exit patterns
            'is_bounce': is_bounce,
            'has_immediate_exit': has_immediate_exit,
            
            # Page value features
            'max_category_value': max_category_value,
            'avg_category_value': avg_category_value,
            'std_category_value': std_category_value,
            'max_product_value': max_product_value,
            'avg_product_value': avg_product_value,
            
            # Temporal intensity
            'session_duration_minutes': session_duration_minutes,
            'max_time_gap': max_time_gap,
            'std_time_between_events': std_time_between_events,
            'avg_time_between_events': avg_time_between_events,
            
            # Engagement depth
            'time_per_product': time_per_product,
            'max_product_interactions': max_product_interactions,
            
            # Exploration patterns
            'exploration_score': exploration_score,
            'category_diversity': category_diversity,
            
            # Advanced behavioral
            'decision_changes': decision_changes,
            'has_multiple_addcart_sequence': has_multiple_addcart_sequence,
            'has_view_addcart_buy_sequence': has_view_addcart_buy_sequence,
            'funnel_addcart_to_buy': funnel_addcart_to_buy,
            
            # High-value indicators
            'has_high_value_category': has_high_value_category,
            'has_ultra_high_value_category': has_ultra_high_value_category,
            
            # Basic features (still important)
            'unique_categories': unique_categories,
            'unique_products': unique_products,
            'total_events': len(group),
            'buy_count': monetary_proxy,
            
            # Target
            'session_value': group.iloc[0].get('session_value', np.nan)
        }