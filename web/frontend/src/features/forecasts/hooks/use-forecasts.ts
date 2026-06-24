import { useQuery } from "@tanstack/react-query";
import { forecastsApi } from "@/features/forecasts/lib/api";
import type { RetrainResponse, RetrainStatusMap } from "@/features/forecasts/types";
import { useMutation } from "@tanstack/react-query";

export function useForecasts(params?: {
  item?: string;
  start_date?: string;
  end_date?: string;
  page?: number;
  page_size?: number;
}) {
  return useQuery({
    queryKey: ["forecasts", params],
    queryFn: () => forecastsApi.list(params),
  });
}

export function useForecastSummary() {
  return useQuery({
    queryKey: ["forecasts", "summary"],
    queryFn: () => forecastsApi.summary(),
  });
}

export function useRetrain() {
  return useMutation<
    RetrainResponse,
    Error,
    { model_type: string; max_items?: number; include_new_products?: boolean; end_date?: string }
  >({
    mutationFn: (params) => forecastsApi.retrain(params),
  });
}

export function useRetrainCancel() {
  return useMutation<{ status: string }, Error, string>({
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
