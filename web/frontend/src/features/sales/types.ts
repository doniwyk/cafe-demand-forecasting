export interface DailySale {
  date: string;
  item: string;
  quantity_sold: number;
  category?: string;
}

export interface DailySalePage {
  data: DailySale[];
  total: number;
  page: number;
  page_size: number;
}

export interface DailyTotalSale {
  date: string;
  quantity: number;
  net_sales: number;
  gross_sales: number;
  unique_items: number;
  unique_categories?: number;
}

export interface DailyCategorySale {
  date: string;
  category: string;
  quantity: number;
  net_sales: number;
  gross_sales: number;
  unique_items: number;
}

export interface ItemInfo {
  name: string;
  category?: string;
}
