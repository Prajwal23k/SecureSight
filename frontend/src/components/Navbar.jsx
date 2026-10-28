
import { Link, NavLink } from "react-router-dom";
import { ShieldCheck } from "lucide-react";

export default function Navbar() {
  return (
    <header className="navbar">
      <Link to="/" className="brand">
        <span className="brand-icon">
          <ShieldCheck size={23} />
        </span>
        <span>
          Secure<span className="accent">Sight</span>
        </span>
      </Link>

      <nav className="nav-links">
        <NavLink to="/" end>
          Home
        </NavLink>
        <NavLink to="/detect">
          Image Detection
        </NavLink>
      </nav>

      <Link to="/detect" className="nav-cta">
        Analyze Image <span>↗</span>
      </Link>
    </header>
  );
}
