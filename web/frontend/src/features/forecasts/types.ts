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
  rmse: number;
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
