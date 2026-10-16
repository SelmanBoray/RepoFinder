import { useState } from "react";

/** A button. */
const Btn = ({ label }: { label: string }) => <button>{label}</button>;

export const Counter = () => {
  const [n, setN] = useState(0);
  const increment = () => setN(n + 1);
  return <Btn label={String(n)} onClick={increment} />;
};

/** The application root. */
export default function App() {
  return (
    <main>
      <Counter />
    </main>
  );
}
