import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";

export function ChartCard({
  title,
  children,
  className,
  ...props
}: {
  title: string;
  children: React.ReactNode;
  className?: string;
} & React.HTMLAttributes<HTMLDivElement>) {
  return (
    <Card className={cn("flex flex-col flex-1 w-full", className)} {...props}>
      <CardHeader className="shrink-0">
        <CardTitle>{title}</CardTitle>
      </CardHeader>
      <CardContent className="flex-1 flex flex-col min-h-0 min-w-0 w-full">{children}</CardContent>
    </Card>
  );
}
