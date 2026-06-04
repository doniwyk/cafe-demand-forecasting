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

export interface ItemInfo {
  name: string;
  category?: string;
}
