import { useQuery } from "@tanstack/react-query";
import { salesApi } from "@/features/sales/lib/api";

export function useDailySales(params?: {
  item?: string;
  start_date?: string;
  end_date?: string;
  page?: number;
  page_size?: number;
}) {
  return useQuery({
    queryKey: ["sales", "daily", params],
    queryFn: () => salesApi.daily(params),
  });
}

export function useDailyTotalSales(params?: {
  start_date?: string;
  end_date?: string;
  page?: number;
  page_size?: number;
}) {
  return useQuery({
    queryKey: ["sales", "total", params],
    queryFn: () => salesApi.dailyTotal(params),
  });
}

export function useCategorySales(params?: {
  category?: string;
  start_date?: string;
  end_date?: string;
  page?: number;
  page_size?: number;
}) {
  return useQuery({
    queryKey: ["sales", "category", params],
    queryFn: () => salesApi.dailyCategory(params),
  });
}

export function useItems() {
  return useQuery({
    queryKey: ["sales", "items"],
    queryFn: () => salesApi.items(),
  });
}

export function useCategories() {
  return useQuery({
    queryKey: ["sales", "categories"],
    queryFn: () => salesApi.categories(),
  });
}
