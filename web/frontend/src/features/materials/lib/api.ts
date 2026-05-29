import { http } from "@/lib/request";
import type { MaterialRequirementPage } from "@/features/materials/types";

export const materialsApi = {
  daily(params?: {
    material?: string;
    start_date?: string;
    end_date?: string;
    page?: number;
    page_size?: number;
  }) {
    return http.get<MaterialRequirementPage>("/materials/daily", params);
  },
  forecast(params?: {
    material?: string;
    start_date?: string;
    end_date?: string;
    page?: number;
    page_size?: number;
    model_type?: string;
  }) {
    return http.get<MaterialRequirementPage>("/materials/forecast", params);
  },
  dailyNeed(params?: {
    start_date?: string;
    end_date?: string;
    page?: number;
    page_size?: number;
    model_type?: string;
  }) {
    return http.get<MaterialRequirementPage>("/materials/daily-forecast", params);
  },
};
