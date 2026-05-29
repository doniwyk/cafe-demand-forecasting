import { useQuery } from "@tanstack/react-query";
import { forecastsApi } from "@/features/forecasts/lib/api";
import type { PredictResponse, RetrainResponse, RetrainStatusMap } from "@/features/forecasts/types";
import { useMutation } from "@tanstack/react-query";

export function useForecasts(params?: {
  item?: string;
  start_date?: string;
  end_date?: string;
  model_type?: string;
  page?: number;
  page_size?: number;
}) {
  return useQuery({
    queryKey: ["forecasts", params],
    queryFn: () => forecastsApi.list(params),
  });
}

export function useForecastSummary(modelType?: string) {
  return useQuery({
    queryKey: ["forecasts", "summary", modelType],
    queryFn: () => forecastsApi.summary(modelType),
  });
}

export function usePredict() {
  return useMutation<
    PredictResponse,
    Error,
    { items: string[]; weeks?: number; model_type?: string }
  >({
    mutationFn: (data) => forecastsApi.predict(data),
  });
}

export function useRetrain() {
  return useMutation<
    RetrainResponse,
    Error,
    { model_type: string; max_items?: number; sync_hus?: boolean; include_new_products?: boolean }
  >({
    mutationFn: (params) => forecastsApi.retrain(params),
  });
}

export function useRetrainCancel() {
  return useMutation<{ status: string; model_type?: string }, Error, string>({
    mutationFn: (modelType) => forecastsApi.retrainCancel(modelType),
  });
}

export function useRetrainStatus() {
  return useQuery<RetrainStatusMap>({
    queryKey: ["forecasts", "retrain-status"],
    queryFn: () => forecastsApi.retrainStatus(),
    refetchInterval: 5_000,
  });
}

export function useCleanup() {
  return useMutation<
    {
      deleted_runs: number;
      deleted_forecasts: number;
      deleted_class_metrics: number;
      deleted_top_items: number;
    },
    Error,
    void
  >({
    mutationFn: () => forecastsApi.cleanup(),
  });
}
