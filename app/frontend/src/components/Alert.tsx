type AlertProps = {
    id: number;
    type: "red" | "yellow" | "green";
    text: string;
    onDelete: (id: number) => void;
  };

export default function Alert({ id, type, text, onDelete }: AlertProps) {
    return (
        <div className={`alert alert-${type}`}>
            <span><strong>Alert&nbsp;:</strong> {text}</span>
            <button onClick={() => onDelete(id)}>×</button>
        </div>
    );
}
