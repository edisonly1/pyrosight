import type { Environment } from "@/lib/types";
import { envItems } from "@/lib/utils";

export default function EnvGrid({ env }: { env: Environment }) {
  return (
    <div className="grid grid-cols-3 gap-3">
      {envItems(env).map(([val, label]) => (
        <div key={label} className="bg-bg-warm rounded-lg p-4 text-center">
          <div className="font-mono text-lg font-semibold">{val}</div>
          <div className="text-[11px] text-text-3 mt-1">{label}</div>
        </div>
      ))}
    </div>
  );
}
