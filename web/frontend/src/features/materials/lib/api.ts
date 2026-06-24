import { http } from "@/lib/request";
import type { MaterialRequirementPage } from "@/features/materials/types";

export const materialsApi = {
  forecast(params?: {
    material?: string;
    start_date?: string;
    end_date?: string;
    page?: number;
    page_size?: number;
  }) {
    return http.get<MaterialRequirementPage>("/materials/forecast", params);
  },
};
