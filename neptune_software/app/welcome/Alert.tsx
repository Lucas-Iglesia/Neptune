type AlertProps = {
    id: number;
    type: "red" | "yellow" | "green";
    text: string;
    onDelete: (id: number) => void;
  };

export default function Alert({ id, type, text, onDelete }: AlertProps) {
    return (
        <div
            className={`flex items-center justify-between border px-4 py-3 rounded
                ${type === "red" ? "bg-red-100 border-red-400 text-red-700" : ""}
                ${type === "yellow" ? "bg-yellow-100 border-yellow-400 text-yellow-700" : ""}
                ${type === "green" ? "bg-green-100 border-green-400 text-green-700" : ""}
            `}
        >
            <div>
                <strong>Alert:</strong> {text}
            </div>
            <button
                onClick={() => onDelete(id)}
                className={`ml-2 font-bold
                ${type === "red" ? "text-red-500 hover:text-red-700" : ""}
                ${type === "yellow" ? "text-yellow-500 hover:text-yellow-700" : ""}
                ${type === "green" ? "text-green-500 hover:text-green-700" : ""}
                `}
            >
                ×
            </button>
        </div>
    );
}
