export default function StatCard({
  value,
  label,
  color,
}: {
  value: string;
  label: string;
  color?: "fire" | "blue" | "green" | "amber";
}) {
  const colorClass = color ? `text-${color}` : "text-text";
  return (
    <div className="text-center">
      <div className={`font-mono text-4xl md:text-5xl font-semibold leading-none ${colorClass}`}>
        {value}
      </div>
      <div className="text-sm text-text-3 mt-2">{label}</div>
    </div>
  );
}
