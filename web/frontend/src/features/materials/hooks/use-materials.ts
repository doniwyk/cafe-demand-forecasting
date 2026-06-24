import { useQuery } from "@tanstack/react-query";
import { materialsApi } from "@/features/materials/lib/api";

export function useDailyNeed(params?: {
  material?: string;
  start_date?: string;
  end_date?: string;
  page?: number;
  page_size?: number;
}) {
  return useQuery({
    queryKey: ["materials", "daily-need", params],
    queryFn: () => materialsApi.forecast(params),
  });
}
