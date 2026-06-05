import { http } from "@/lib/request";
import type { DailySalePage, ItemInfo } from "@/features/sales/types";

export const salesApi = {
  daily(params?: {
    item?: string;
    start_date?: string;
    end_date?: string;
    page?: number;
    page_size?: number;
  }) {
    return http.get<DailySalePage>("/sales/daily", params);
  },
  items() {
    return http.get<ItemInfo[]>("/sales/items");
  },
  categories() {
    return http.get<string[]>("/sales/categories");
  },
  latestDate() {
    return http.get<string | null>("/sales/latest-date");
  },
};
