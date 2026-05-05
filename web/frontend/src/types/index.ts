export interface DailySale {
  date: string;
  item: string;
  quantity_sold: number;
  category?: string;
}

export interface DailySalePage {
  data: DailySale[];
  total: number;
  page: number;
  page_size: number;
}

export interface DailyTotalSale {
  date: string;
  quantity: number;
  net_sales: number;
  gross_sales: number;
  unique_items: number;
  unique_categories?: number;
}

export interface DailyCategorySale {
  date: string;
  category: string;
  quantity: number;
  net_sales: number;
  gross_sales: number;
  unique_items: number;
}

export interface ItemInfo {
  name: string;
  category?: string;
}

export interface ForecastRecord {
  date: string;
  item: string;
  quantity_sold: number;
}

export interface ForecastPage {
  data: ForecastRecord[];
  total: number;
  page: number;
  page_size: number;
}

export interface ModelMetrics {
  r2: number;
  wmape: number;
  mae: number;
  median_period_accuracy: number;
  periods_within_20pct: number;
  periods_within_50pct: number;
}

export interface ClassMetrics {
  n_items: number;
  wmape: number;
  median_period_accuracy: number;
}

export interface PredictResponse {
  data: ForecastRecord[];
  total: number;
}

export interface RetrainResponse {
  status: string;
  message: string;
}

export interface RetrainStatusEntry {
  status: string;
  message: string;
  logs: string[];
}

export type RetrainStatusMap = Record<string, RetrainStatusEntry>;

export interface TopItem {
  item: string;
  quantity_sold: number;
  predicted: number;
  accuracy_pct: number;
}

export interface ForecastSummary {
  global_metrics: ModelMetrics;
  class_metrics: Record<string, ClassMetrics>;
  top_items: TopItem[];
}

export interface DailyMaterialRequirement {
  date: string;
  raw_material: string;
  quantity_required: number;
}

export interface MaterialRequirementPage {
  data: DailyMaterialRequirement[];
  total: number;
  page: number;
  page_size: number;
}

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
