import { CheckIcon, LoaderIcon, AlertCircleIcon } from "lucide-react";

export function StatusIcon({ status }: { status: string }) {
  switch (status) {
    case "training":
      return <LoaderIcon className="size-4 animate-spin" />;
    case "success":
      return <CheckIcon className="size-4 text-emerald-600" />;
    case "error":
      return <AlertCircleIcon className="size-4 text-destructive" />;
    default:
      return null;
  }
}
