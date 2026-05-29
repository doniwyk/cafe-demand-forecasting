export interface DailyMaterialRequirement {
  date: string;
  raw_material: string;
  quantity_required: number;
  unit?: string;
}

export interface MaterialRequirementPage {
  data: DailyMaterialRequirement[];
  total: number;
  page: number;
  page_size: number;
}
