import { init } from "./app";

import "./reset.css";
import "./style.css";

const container = document.querySelector<HTMLDivElement>("#app")!;
const errorContainer = document.querySelector<HTMLDivElement>("#app-error")!;

init({ container }).catch((e) => {
  errorContainer.innerText = e.message;
  errorContainer.style.display = "flex";
});
