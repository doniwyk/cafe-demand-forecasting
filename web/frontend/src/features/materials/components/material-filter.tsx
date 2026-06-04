import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
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
import { format, addDays } from "date-fns";
import { ChevronsUpDownIcon, CalendarIcon } from "lucide-react";
import { useTranslation } from "react-i18next";

interface MaterialFilterProps {
  materials: string[];
  selectedMaterial: string | null;
  onSelectMaterial: (material: string | null) => void;
  dateRange: { from?: Date; to?: Date };
  onDateRangeChange: (range: { from?: Date; to?: Date }) => void;
}

export function MaterialFilter({
  materials,
  selectedMaterial,
  onSelectMaterial,
  dateRange,
  onDateRangeChange,
}: MaterialFilterProps) {
  const [materialOpen, setMaterialOpen] = useState(false);
  const [dateOpen, setDateOpen] = useState(false);
  const { t } = useTranslation();

  return (
    <Card data-tour="material-filter">
      <CardHeader>
        <CardTitle>{t("materials.materialFilter")}</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="flex gap-4 flex-wrap items-end">
          <div className="flex flex-col gap-1.5">
            <label className="text-sm font-medium">{t("materials.dateRange")}</label>
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
                        {t("materials.selectDateRange")}
                      </span>
                    )}
                  </Button>
                }
              />
              <PopoverContent className="w-auto p-0" align="start">
                <Calendar
                  mode="range"
                  selected={dateRange as { from: Date; to?: Date }}
                  onSelect={(range) =>
                    onDateRangeChange(range ?? { from: new Date(), to: addDays(new Date(), 7) })
                  }
                  numberOfMonths={2}
                />
              </PopoverContent>
            </Popover>
          </div>
          <div className="flex flex-col gap-1.5">
            <label className="text-sm font-medium">{t("materials.material")}</label>
            <Popover open={materialOpen} onOpenChange={setMaterialOpen}>
              <PopoverTrigger
                render={
                  <Button
                    variant="outline"
                    role="combobox"
                    aria-expanded={materialOpen}
                    className="w-70 justify-between"
                  />
                }
              >
                {selectedMaterial || t("materials.selectMaterial")}
                <ChevronsUpDownIcon className="ml-auto size-4 shrink-0 opacity-50" />
              </PopoverTrigger>
              <PopoverContent className="w-70 p-0" align="start">
                <Command>
                  <CommandInput placeholder={t("materials.searchMaterials")} />
                  <CommandList>
                    <CommandEmpty>{t("materials.noMaterialsFound")}</CommandEmpty>
                    <CommandGroup>
                      {materials.map((mat) => (
                        <CommandItem
                          key={mat}
                          value={mat}
                          onSelect={(value) => {
                            onSelectMaterial(value === selectedMaterial ? null : value);
                            setMaterialOpen(false);
                          }}
                        >
                          {mat}
                        </CommandItem>
                      ))}
                    </CommandGroup>
                  </CommandList>
                </Command>
              </PopoverContent>
            </Popover>
          </div>
          {(selectedMaterial || dateRange.from || dateRange.to) && (
            <Button
              variant="ghost"
              size="sm"
              onClick={() => {
                onSelectMaterial(null);
                onDateRangeChange({ from: new Date(), to: addDays(new Date(), 7) });
              }}
            >
              {t("common.clear")}
            </Button>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
