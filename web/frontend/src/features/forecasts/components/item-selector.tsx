import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import {
  Command,
  CommandInput,
  CommandList,
  CommandEmpty,
  CommandGroup,
  CommandItem,
} from "@/components/ui/command";
import { Calendar } from "@/components/ui/calendar";
import { useItems } from "@/features/sales/hooks/use-sales";
import { format, addDays } from "date-fns";
import { ChevronsUpDownIcon, CalendarIcon } from "lucide-react";
import { useTranslation } from "react-i18next";

interface ItemSelectorProps {
  selectedItem: string | null;
  onSelectItem: (item: string | null) => void;
  dateRange: { from?: Date; to?: Date };
  onDateRangeChange: (range: { from?: Date; to?: Date }) => void;
}

export function ItemSelector({
  selectedItem,
  onSelectItem,
  dateRange,
  onDateRangeChange,
}: ItemSelectorProps) {
  const [open, setOpen] = useState(false);
  const [dateOpen, setDateOpen] = useState(false);
  const items = useItems();
  const { t } = useTranslation();

  return (
    <Card data-tour="item-selector">
      <CardHeader>
        <CardTitle>{t("forecasts.itemSelector")}</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="flex gap-4 flex-wrap items-end">
          <div className="flex flex-col gap-1.5">
            <label className="text-sm font-medium">{t("forecasts.item")}</label>
            <Popover open={open} onOpenChange={setOpen}>
              <PopoverTrigger
                render={
                  <Button
                    variant="outline"
                    role="combobox"
                    aria-expanded={open}
                    className="w-70 justify-between"
                  />
                }
              >
                {selectedItem || t("forecasts.selectItem")}
                <ChevronsUpDownIcon className="ml-auto size-4 shrink-0 opacity-50" />
              </PopoverTrigger>
              <PopoverContent className="w-70 p-0" align="start">
                <Command>
                  <CommandInput placeholder={t("forecasts.searchItems")} />
                  <CommandList>
                    <CommandEmpty>{t("forecasts.noItemsFound")}</CommandEmpty>
                    <CommandGroup>
                      {items.data?.map((item) => (
                        <CommandItem
                          key={item.name}
                          value={item.name}
                          onSelect={(value) => {
                            onSelectItem(value === selectedItem ? null : value);
                            setOpen(false);
                          }}
                        >
                          {item.name}
                        </CommandItem>
                      ))}
                    </CommandGroup>
                  </CommandList>
                </Command>
              </PopoverContent>
            </Popover>
          </div>

          <div className="flex flex-col gap-1.5">
            <label className="text-sm font-medium">{t("forecasts.dateRange")}</label>
            <Popover open={dateOpen} onOpenChange={setDateOpen}>
              <PopoverTrigger
                render={
                  <Button variant="outline" className="w-70 justify-start text-left font-normal">
                    <CalendarIcon className="mr-2 size-4 shrink-0" />
                    {dateRange.from ? (
                      dateRange.to ? (
                        <>
                          {format(dateRange.from, "MMM dd, yyyy")} —{" "}
                          {format(dateRange.to, "MMM dd, yyyy")}
                        </>
                      ) : (
                        format(dateRange.from, "MMM dd, yyyy")
                      )
                    ) : (
                      <span className="text-muted-foreground">
                        {t("forecasts.selectDateRange")}
                      </span>
                    )}
                  </Button>
                }
              />
              <PopoverContent className="w-auto p-0" align="start">
                <Calendar
                  mode="range"
                  selected={dateRange as { from: Date; to?: Date }}
                  onSelect={(range) => onDateRangeChange(range ?? {})}
                  numberOfMonths={2}
                />
              </PopoverContent>
            </Popover>
          </div>

          {(selectedItem || dateRange.from || dateRange.to) && (
            <Button
              variant="ghost"
              size="sm"
              onClick={() => {
                onSelectItem(null);
                onDateRangeChange({});
              }}
            >
              {t("common.clear")}
            </Button>
          )}
        </div>
        {selectedItem && (
          <div className="mt-3 flex items-center gap-2">
            <Badge variant="secondary">{selectedItem}</Badge>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
