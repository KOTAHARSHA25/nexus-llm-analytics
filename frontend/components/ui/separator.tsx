import * as React from "react";
import { cn } from "../../lib/utils";

export interface SeparatorProps extends React.HTMLAttributes<HTMLDivElement> {
  orientation?: "horizontal" | "vertical";
}

export const Separator = React.forwardRef<HTMLDivElement, SeparatorProps>(
  ({ className, orientation = "horizontal", ...props }, ref) => (
    <div
      ref={ref}
      role="separator"
      {...(orientation === "vertical"
        ? { 'aria-orientation': 'vertical' as const }
        : {})}
      className={cn(
        orientation === "horizontal"
          ? "h-px w-full bg-gray-200"
          : "h-full w-px bg-gray-200",
        className
      )}
      {...props}
    />
  )
);
Separator.displayName = "Separator";

export default Separator;
