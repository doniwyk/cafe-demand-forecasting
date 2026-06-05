import { useQuery } from "@tanstack/react-query";
import { salesApi } from "@/features/sales/lib/api";

export function useItems() {
  return useQuery({
    queryKey: ["sales", "items"],
    queryFn: () => salesApi.items(),
  });
}

export function useLatestSalesDate() {
  return useQuery({
    queryKey: ["sales", "latest-date"],
    queryFn: () => salesApi.latestDate(),
  });
}
