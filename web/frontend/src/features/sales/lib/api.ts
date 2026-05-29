import { http } from "@/lib/request";
import type { DailySalePage, DailyTotalSale, DailyCategorySale, ItemInfo } from "@/features/sales/types";

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
  dailyTotal(params?: {
    start_date?: string;
    end_date?: string;
    page?: number;
    page_size?: number;
  }) {
    return http.get<DailyTotalSale[]>("/sales/daily/total", params);
  },
  dailyCategory(params?: {
    category?: string;
    start_date?: string;
    end_date?: string;
    page?: number;
    page_size?: number;
  }) {
    return http.get<DailyCategorySale[]>("/sales/daily/category", params);
  },
  items() {
    return http.get<ItemInfo[]>("/sales/items");
  },
  categories() {
    return http.get<string[]>("/sales/categories");
  },
};
