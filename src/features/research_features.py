import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class ResearchBasedFeatureEngine:
    """Research-based features WITHOUT data leakage"""
    
    def __init__(self):
        self.category_buy_rates: Dict[str, float] = {}
        self.product_buy_rates: Dict[str, float] = {}
        self.category_engagement_scores: Dict[str, float] = {}
        
    def fit_category_values(self, df: pd.DataFrame) -> 'ResearchBasedFeatureEngine':
        """Learn behavioral patterns WITHOUT using session_value"""
        logger.info("Learning behavioral patterns without data leakage...")
        
        # Calculate category buy rates (behavioral proxy)
        category_stats = df.groupby('category_id').agg({
            'event_type': lambda x: (x == 'BUY').sum(),
            'user_session': 'nunique'
        }).reset_index()
        
        category_stats['buy_rate'] = category_stats['event_type'] / category_stats['user_session']
        self.category_buy_rates = dict(zip(category_stats['category_id'], category_stats['buy_rate']))
        
        # Product buy rates
        product_stats = df.groupby('product_id').agg({
            'event_type': lambda x: (x == 'BUY').sum(),
            'user_session': 'nunique'
        }).reset_index()
        
        product_stats['buy_rate'] = product_stats['event_type'] / product_stats['user_session']
        self.product_buy_rates = dict(zip(product_stats['product_id'], product_stats['buy_rate']))
        
        # Category engagement (events per session)
        category_engagement = df.groupby('category_id').agg({
            'event_type': 'count',
            'user_session': 'nunique'
        }).reset_index()
        
        category_engagement['engagement'] = category_engagement['event_type'] / category_engagement['user_session']
        self.category_engagement_scores = dict(zip(category_engagement['category_id'], category_engagement['engagement']))
        
        logger.info(f"Learned patterns for {len(self.category_buy_rates)} categories")
        return self
    
    def create_research_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create research-based features without data leakage"""
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
        """Extract features without using session_value"""
        
        events_sequence = group['event_type'].tolist()
        events_str = '->'.join(events_sequence)
        
        # 1. MONETARY PROXY (buy count - no leakage)
        monetary_proxy = events_str.count('BUY')
        
        # 2. BEHAVIORAL VALUE PROXY (based on category/product buy rates)
        categories = group['category_id'].unique()
        products = group['product_id'].unique()
        
        if len(categories) > 0:
            category_scores = [self.category_buy_rates.get(cat, 0) for cat in categories]
            total_category_score = sum(category_scores)
            max_category_score = max(category_scores)
            avg_category_score = np.mean(category_scores)
        else:
            total_category_score = max_category_score = avg_category_score = 0
            
        if len(products) > 0:
            product_scores = [self.product_buy_rates.get(prod, 0) for prod in products]
            max_product_score = max(product_scores)
            avg_product_score = np.mean(product_scores)
        else:
            max_product_score = avg_product_score = 0
        
        # 3. ENGAGEMENT PROXY
        if len(categories) > 0:
            engagement_scores = [self.category_engagement_scores.get(cat, 0) for cat in categories]
            avg_engagement = np.mean(engagement_scores)
            max_engagement = max(engagement_scores)
        else:
            avg_engagement = max_engagement = 0
        
        # 4. DECISIVENESS SCORE (no leakage)
        if 'BUY' in events_sequence:
            first_buy_position = events_sequence.index('BUY')
            decisiveness_score = 1.0 - (first_buy_position / len(events_sequence))
        else:
            decisiveness_score = 0.0
        
        # 5. BEHAVIORAL PATTERNS (no leakage)
        has_comparison_pattern = int(len(set(group['category_id'])) > 2)
        is_bounce = int(len(group) == 1)
        has_immediate_exit = int(len(group) == 1 and group.iloc[0]['event_type'] == 'VIEW')
        
        # 6. SESSION INTENSITY
        if len(group) > 1:
            session_duration_minutes = (group['event_time'].max() - group['event_time'].min()).total_seconds() / 60
            time_diffs = group['event_time'].diff().dt.total_seconds().dropna()
            
            max_time_gap = time_diffs.max()
            std_time_between_events = time_diffs.std() if len(time_diffs) > 1 else 0
        else:
            session_duration_minutes = 0
            max_time_gap = 0
            std_time_between_events = 0
        
        # 7. PRODUCT ENGAGEMENT
        products_viewed = group[group['event_type'] == 'VIEW']['product_id'].nunique()
        if products_viewed > 0 and session_duration_minutes > 0:
            time_per_product = session_duration_minutes / products_viewed
        else:
            time_per_product = 0
            
        product_interactions = group['product_id'].value_counts()
        max_product_interactions = product_interactions.max() if len(product_interactions) > 0 else 0
        
        # 8. EXPLORATION PATTERNS
        unique_categories = group['category_id'].nunique()
        unique_products = group['product_id'].nunique()
        
        exploration_score = (unique_categories + unique_products) / len(group)
        category_diversity = unique_categories / max(unique_products, 1)
        
        # 9. BEHAVIORAL PATTERNS
        decision_changes = events_str.count('ADD_CART->REMOVE_CART')
        has_multiple_addcart_sequence = int(events_str.count('ADD_CART') >= 2)
        has_view_addcart_buy_sequence = int('VIEW->ADD_CART->BUY' in events_str)
        funnel_addcart_to_buy = events_str.count('ADD_CART->BUY') / max(events_str.count('ADD_CART'), 1)
        
        # 10. HIGH-VALUE BEHAVIORAL INDICATORS (based on buy rates, not session_value)
        high_buy_rate_threshold = np.percentile(list(self.category_buy_rates.values()), 75)
        ultra_high_buy_rate_threshold = np.percentile(list(self.category_buy_rates.values()), 90)
        
        has_high_buy_rate_category = int(max_category_score > high_buy_rate_threshold)
        has_ultra_high_buy_rate_category = int(max_category_score > ultra_high_buy_rate_threshold)
        
        return {
            'user_session': session_id,
            
            # Behavioral proxies (no leakage)
            'monetary_proxy': monetary_proxy,
            'total_category_score': total_category_score,
            'decisiveness_score': decisiveness_score,
            'has_comparison_pattern': has_comparison_pattern,
            
            # Behavioral value proxies
            'max_category_score': max_category_score,
            'avg_category_score': avg_category_score,
            'max_product_score': max_product_score,
            'avg_product_score': avg_product_score,
            
            # Engagement proxies
            'avg_engagement': avg_engagement,
            'max_engagement': max_engagement,
            
            # Session patterns
            'is_bounce': is_bounce,
            'has_immediate_exit': has_immediate_exit,
            'session_duration_minutes': session_duration_minutes,
            'max_time_gap': max_time_gap,
            'std_time_between_events': std_time_between_events,
            
            # Engagement depth
            'time_per_product': time_per_product,
            'max_product_interactions': max_product_interactions,
            
            # Exploration
            'exploration_score': exploration_score,
            'category_diversity': category_diversity,
            
            # Advanced behavioral
            'decision_changes': decision_changes,
            'has_multiple_addcart_sequence': has_multiple_addcart_sequence,
            'has_view_addcart_buy_sequence': has_view_addcart_buy_sequence,
            'funnel_addcart_to_buy': funnel_addcart_to_buy,
            
            # Buy rate indicators
            'has_high_buy_rate_category': has_high_buy_rate_category,
            'has_ultra_high_buy_rate_category': has_ultra_high_buy_rate_category,
            
            # Basic counts
            'unique_categories': unique_categories,
            'unique_products': unique_products,
            'total_events': len(group),
            'buy_count': monetary_proxy,
            
            # Target
            'session_value': group.iloc[0].get('session_value', np.nan)
        }
    
    def save(self, filepath: str):
        """Save feature engine"""
        import joblib
        joblib.dump(self, filepath)