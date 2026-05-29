import { Card, CardContent, CardDescription, CardHeader } from "@/components/ui/card";

function CardAction({ icon }: { icon: React.ReactNode }) {
  return (
    <div className="col-start-2 row-span-2 row-start-1 self-start justify-self-end text-muted-foreground">
      {icon}
    </div>
  );
}

export function KpiCard({
  title,
  value,
  description,
  icon,
}: {
  title: string;
  value: string | number;
  description: string;
  icon: React.ReactNode;
}) {
  return (
    <Card>
      <CardHeader>
        <CardDescription>{title}</CardDescription>
        <CardAction icon={icon} />
      </CardHeader>
      <CardContent>
        <div className="text-2xl font-bold tracking-tight">{value}</div>
        <p className="text-xs text-muted-foreground mt-1">{description}</p>
      </CardContent>
    </Card>
  );
}
