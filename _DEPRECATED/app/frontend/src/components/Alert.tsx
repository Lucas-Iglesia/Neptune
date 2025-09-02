import { useEffect, useRef } from "react";

type AlertProps = {
    id: number;
    type: "red" | "yellow" | "green";
    text: string;
    playAudio?: boolean;
    onDelete: (id: number) => void;
  };

export default function Alert({ id, type, text, playAudio, onDelete }: AlertProps) {
    const audioRef = useRef<HTMLAudioElement>(null);

    useEffect(() => {
        if (playAudio && audioRef.current) {
            audioRef.current.play().catch(error => {
                console.error("Error playing alert audio:", error);
            });
        }
    }, [playAudio]);

    return (
        <div className={`alert alert-${type}`}>
            <span><strong>Alert&nbsp;:</strong> {text}</span>
            <button onClick={() => onDelete(id)}>×</button>
            {playAudio && (
                <audio ref={audioRef} preload="auto">
                    <source src="http://localhost:5000/api/alert-audio" type="audio/mpeg" />
                </audio>
            )}
        </div>
    );
}
