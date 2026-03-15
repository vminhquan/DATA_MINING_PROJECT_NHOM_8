import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

class AssociationMiner:
    def __init__(self, min_support=0.01, min_confidence=0.1):
        self.min_support = min_support
        self.min_confidence = min_confidence

    def mine_rules(self, df_encoded):
        frequent_itemsets = apriori(df_encoded, min_support=self.min_support, use_colnames=True)
        if frequent_itemsets.empty:
            return pd.DataFrame()
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=self.min_confidence)
        return rules.sort_values(by=['lift'], ascending=False)