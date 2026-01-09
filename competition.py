"""
Competitive Learning Module
Strengthening one category weakens competing categories
"""
import numpy as np
from collections import defaultdict
from config import COMPETITION_STRENGTH


class CompetitionManager:
    
    def __init__(self, knowledge_graph, competition_strength=COMPETITION_STRENGTH):
        self.kg = knowledge_graph
        self.competition_strength = competition_strength
        self.competition_matrix = {}
        self._build_competition_matrix()
    
    def _build_competition_matrix(self):
        """Calculate competition between categories from co-occurrence data"""
        categories = self.kg.categories
        co_occurrence = defaultdict(lambda: defaultdict(int))
        category_counts = defaultdict(int)
        
        for search in self.kg.search_history:
            cats = list(search.get('categories', {}).keys())
            for cat in cats:
                category_counts[cat] += 1
                for other_cat in cats:
                    if cat != other_cat:
                        co_occurrence[cat][other_cat] += 1
        
        for cat_a in categories:
            self.competition_matrix[cat_a] = {}
            for cat_b in categories:
                if cat_a == cat_b:
                    self.competition_matrix[cat_a][cat_b] = 0.0
                else:
                    co_count = co_occurrence[cat_a][cat_b]
                    total_a = category_counts[cat_a]
                    
                    if total_a > 0:
                        co_occurrence_rate = co_count / total_a
                        competition = 1.0 - co_occurrence_rate
                        self.competition_matrix[cat_a][cat_b] = np.clip(competition, 0.0, 1.0)
                    else:
                        self.competition_matrix[cat_a][cat_b] = 0.5
    
    def get_competition(self, cat_a, cat_b):
        """Get competition strength between two categories"""
        if cat_a not in self.competition_matrix or cat_b not in self.competition_matrix[cat_a]:
            return 0.5
        return self.competition_matrix[cat_a][cat_b]
    
    def apply_competition(self, strengthened_category, learning_rate):
        """When category is strengthened, weaken competitors"""
        if not self.kg.G.has_edge("USER", strengthened_category):
            return
        
        for competitor in self.kg.categories:
            if competitor == strengthened_category:
                continue
            
            if not self.kg.G.has_edge("USER", competitor):
                continue
            
            competition = self.get_competition(strengthened_category, competitor)
            
            if competition > 0.5:
                current_weight = self.kg.G["USER"][competitor]['weight']
                penalty = self.competition_strength * learning_rate * competition
                new_weight = max(current_weight - penalty, 0.0)
                self.kg.G["USER"][competitor]['weight'] = new_weight
                
                if self.kg.G.has_edge(competitor, "USER"):
                    current_bwd = self.kg.G[competitor]["USER"]['weight']
                    new_bwd = max(current_bwd - penalty * 0.5, 0.0)
                    self.kg.G[competitor]["USER"]['weight'] = new_bwd
    
    def rebuild_competition_matrix(self):
        """Rebuild after significant graph changes"""
        self.competition_matrix = {}
        self._build_competition_matrix()