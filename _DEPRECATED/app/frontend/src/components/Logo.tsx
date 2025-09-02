import logo from "../assets/neptune_logo.png";

export default function Logo() {
  return (
    <>
      <img src={logo} alt="Logo Neptune" className="logo" />
      <span className="brand">Neptune</span>
    </>
  );
}
