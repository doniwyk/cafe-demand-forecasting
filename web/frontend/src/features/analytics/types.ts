export interface ABCItem {
  item: string;
  vol: number;
  cum: number;
  pct: number;
  class_label: string;
}

export interface ABCAnalysisResponse {
  class_metrics: Record<string, { n_items: number; total_volume: number; pct_volume: number }>;
  classifications: ABCItem[];
}

export interface AssociationRule {
  antecedents: string;
  consequents: string;
  support: number;
  confidence: number;
  lift: number;
}

export interface TopSellingItem {
  item: string;
  total_quantity: number;
}
