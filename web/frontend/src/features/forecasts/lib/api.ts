import { http } from "@/lib/request";
import type {
  ForecastPage,
  ForecastSummary,
  PredictResponse,
  RetrainResponse,
  RetrainStatusMap,
} from "@/features/forecasts/types";

export const forecastsApi = {
  list(params?: {
    item?: string;
    start_date?: string;
    end_date?: string;
    page?: number;
    page_size?: number;
  }) {
    return http.get<ForecastPage>("/forecasts", params);
  },
  summary() {
    return http.get<ForecastSummary>("/forecasts/summary");
  },
  predict(data: { items: string[]; weeks?: number }) {
    return http.post<PredictResponse>("/forecasts/predict", data);
  },
  retrain(params: {
    model_type: string;
    max_items?: number;
    include_new_products?: boolean;
    end_date?: string;
  }) {
    return http.post<RetrainResponse>("/forecasts/retrain", params);
  },
  retrainCancel(model_type: string) {
    return http.post<{ status: string; message?: string }>(
      "/forecasts/retrain/cancel",
      { model_type },
    );
  },
  retrainStatus() {
    return http.get<RetrainStatusMap>("/forecasts/retrain/status", { tail: 200 });
  },
  cleanup() {
    return http.post<{
      deleted_runs: number;
      deleted_forecasts: number;
      deleted_class_metrics: number;
      deleted_top_items: number;
    }>("/forecasts/cleanup");
  },
};
