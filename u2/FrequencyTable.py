from math import isnan
from typing import Union, List, Optional, Dict, Tuple, Any

# Activa soporte de gráficos con matplotlib
try:
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    _MATPLOTLIB_AVAILABLE = True
except Exception:
    _MATPLOTLIB_AVAILABLE = False

Number = Union[int, float]


def _is_number(x: Any) -> bool:
    if isinstance(x, bool):
        return False
    if isinstance(x, (int, float)):
        return not (isinstance(x, float) and isnan(x))
    return False


def _percentile(sorted_vals: List[float], q: float) -> float:
    """
    Percentil lineal (0-100). sorted_vals debe estar ordenada ascendente.
    Implementación similar a numpy.percentile con interpolación lineal.
    """
    n = len(sorted_vals)
    if n == 0:
        return float("nan")
    if q <= 0:
        return float(sorted_vals[0])
    if q >= 100:
        return float(sorted_vals[-1])
    pos = (q / 100) * (n - 1)
    i = int(pos)
    frac = pos - i
    if i + 1 < n:
        return float(sorted_vals[i] * (1 - frac) + sorted_vals[i + 1] * frac)
    return float(sorted_vals[i])


class FrequencyTable:
    """
    Tabla de frecuencias sin pandas/NumPy.
    - data: lista
    - bins: None, int, o lista de cortes (para datos numéricos)
    - bin_labels: etiquetas opcionales para cortes
    La tabla se almacena como vectores (listas) y 'categories' alineada.
    """

    def __init__(self, data: List[Any],
                 bins: Optional[Union[int, List[Number]]] = None,
                 bin_labels: Optional[List[str]] = None):
        self.data = list(data)
        self.bins = bins
        self.bin_labels = bin_labels

        self.processed = self._process()
        self.table = self._frequencies()
        self.stats = self._statistics()

    def _process(self) -> List[Any]:
        # Filtrar None/NaN
        clean: List[Any] = []
        for x in self.data:
            if x is None:
                continue
            if isinstance(x, float) and isnan(x):
                continue
            clean.append(x)

        if self.bins is None:
            return clean

        # Discretización numérica
        values: List[float] = [float(x) for x in clean if _is_number(x)]
        if not values:
            return []

        if isinstance(self.bins, int):
            nbins = max(1, self.bins)
            vmin, vmax = min(values), max(values)
            if vmin == vmax:
                step = 1.0 / nbins
                edges = [vmin - 0.5 + i * step for i in range(nbins + 1)]
            else:
                step = (vmax - vmin) / nbins
                edges = [vmin + i * step for i in range(nbins + 1)]
        else:
            edges = [float(b) for b in self.bins]
            if len(edges) < 2:
                raise ValueError("bins como lista debe tener al menos 2 bordes")

        # Etiquetas
        if self.bin_labels is not None:
            if len(self.bin_labels) != len(edges) - 1:
                raise ValueError("bin_labels debe tener len == len(edges)-1")
            labels = list(self.bin_labels)
        else:
            labels = [f"[{edges[i]:.3g}, {edges[i + 1]:.3g})" for i in range(len(edges) - 1)]

        # Asignación a bins (incluye borde derecho del último intervalo)
        def assign_bin(v: float) -> int:
            # búsqueda lineal (suficiente para tamaños moderados)
            for i in range(len(edges) - 1):
                left, right = edges[i], edges[i + 1]
                if (v >= left and v < right) or (i == len(edges) - 2 and v == edges[-1]):
                    return i
            # fuera de rango: clamp
            if v < edges[0]:
                return 0
            return len(edges) - 2

        assigned: List[str] = [labels[assign_bin(v)] for v in values]
        return assigned

    def _frequencies(self) -> Dict[str, List[Any]]:
        if not self.processed:
            return {
                "categories": [],
                "Frequency": [],
                "Relative": [],
                "Percentage": [],
                "Cum_Freq": [],
                "Cum_%": [],
            }

        # Conteo por categorías
        counts: Dict[str, int] = {}
        for c in self.processed:
            key = str(c)
            counts[key] = counts.get(key, 0) + 1

        # Orden por nombre de categoría
        categories = sorted(counts.keys(), key=lambda x: x)
        freq = [counts[c] for c in categories]
        total = sum(freq)
        rel = [(f / total) if total > 0 else 0.0 for f in freq]
        perc = [round(r * 100.0, 2) for r in rel]

        cum_f = []
        s = 0
        for f in freq:
            s += f
            cum_f.append(s)

        cum_p = []
        sp = 0.0
        for p in perc:
            sp += p
            cum_p.append(round(sp, 2))

        return {
            "categories": categories,
            "Frequency": [int(f) for f in freq],
            "Relative": [round(r, 6) for r in rel],
            "Percentage": perc,
            "Cum_Freq": [int(x) for x in cum_f],
            "Cum_%": cum_p,
        }

    def _statistics(self) -> Dict[str, Any]:
        stats: Dict[str, Any] = {
            "observations": len(self.data),
            "unique": len(self.table["categories"])
        }

        # Moda categórica
        if self.table["categories"]:
            max_idx = max(range(len(self.table["Frequency"])), key=lambda i: self.table["Frequency"][i])
            stats.update({
                "mode": self.table["categories"][max_idx],
                "max_freq": self.table["Frequency"][max_idx],
                "min_freq": min(self.table["Frequency"])
            })
        else:
            stats.update({"mode": "N/A", "max_freq": 0, "min_freq": 0})

        # Estadísticas numéricas si aplica
        numeric_vals = [float(x) for x in self.data if _is_number(x)]
        if numeric_vals:
            n = len(numeric_vals)
            sorted_vals = sorted(numeric_vals)
            mean = sum(numeric_vals) / n
            median = (sorted_vals[n // 2] if n % 2 == 1
                      else (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2.0)
            # moda de valores
            val_counts: Dict[float, int] = {}
            for v in numeric_vals:
                val_counts[v] = val_counts.get(v, 0) + 1
            mode_val = max(val_counts.keys(), key=lambda k: val_counts[k])

            # Varianza y desviación estándar (muestral, ddof=1)
            if n > 1:
                var = sum((v - mean) ** 2 for v in numeric_vals) / (n - 1)
                std = var ** 0.5
            else:
                var = 0.0
                std = 0.0
            rng = max(numeric_vals) - min(numeric_vals)
            q1 = _percentile(sorted_vals, 25.0)
            q3 = _percentile(sorted_vals, 75.0)
            iqr = q3 - q1
            coef_var = (std / mean) if mean != 0 else float("nan")

            # Curtosis de Fisher: m4/m2^2 - 3 (bias no corregido)
            m2 = sum((v - mean) ** 2 for v in numeric_vals) / n
            m4 = sum((v - mean) ** 4 for v in numeric_vals) / n
            kurtosis = (m4 / (m2 ** 2) - 3.0) if m2 != 0 else float("nan")

            stats.update({
                "mean": mean,
                "median": median,
                "mode_val": mode_val,
                "variance": var,
                "std_dev": std,
                "range": rng,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "coef_var": coef_var,
                "kurtosis": kurtosis
            })

        return stats

    def summary(self) -> None:
        print("=" * 40)
        print("RESUMEN TABLA DE FRECUENCIAS")
        print("=" * 40)
        for k, v in self.stats.items():
            print(f"{k}: {v}")
        print("\nTabla (vectores):")
        for key in ["categories", "Frequency", "Relative", "Percentage", "Cum_Freq", "Cum_%"]:
            print(f"{key}: {self.table[key]}")

    def get_table_vectors(self) -> Tuple[List[str], Dict[str, List[Any]]]:
        """
        Retorna (categories, columnas) con vectores alineados.
        """
        cats = self.table["categories"]
        cols = {k: v for k, v in self.table.items() if k != "categories"}
        return cats, cols

    def plot_data(self, kind: str = "bar", bins: Optional[int] = None) -> Dict[str, Any]:
        """
        Prepara datos listos para graficar con matplotlib.
        """
        if kind == "hist" and self.bins is None:
            numeric_vals = [float(x) for x in self.data if _is_number(x)]
            return {"x": numeric_vals, "kind": "hist", "bins": bins or 10}

        return {
            "category": self.table["categories"],
            "frequency": self.table["Frequency"],
            "kind": "bar"
        }

    def plot(self, kind: str = "bar", title: Optional[str] = None,
             figsize: Tuple[int, int] = (10, 6), rotation: int = 45,
             color: str = "steelblue", show: bool = True):
        """
        Grafica usando matplotlib.

        Args:
            kind: Tipo de gráfico ("bar" o "hist")
            title: Título del gráfico
            figsize: Tamaño de la figura (ancho, alto)
            rotation: Rotación de etiquetas del eje x
            color: Color de las barras
            show: Si mostrar el gráfico automáticamente
        """
        if not _MATPLOTLIB_AVAILABLE:
            raise RuntimeError("matplotlib no está disponible en el entorno.")

        pdata = self.plot_data(kind=kind)
        title = title or f"Distribución de Frecuencia ({kind})"

        fig, ax = plt.subplots(figsize=figsize)

        if pdata.get("kind") == "hist":
            ax.hist(pdata["x"], bins=pdata["bins"], color=color, alpha=0.7, edgecolor='black')
            ax.set_xlabel("Valores")
            ax.set_ylabel("Frecuencia")
        else:
            # Gráfico de barras
            categories = pdata["category"]
            frequencies = pdata["frequency"]

            # Crear gráfico de barras horizontal si hay muchas categorías o nombres largos
            if len(categories) > 10 or any(len(str(cat)) > 10 for cat in categories):
                ax.barh(range(len(categories)), frequencies, color=color, alpha=0.7)
                ax.set_yticks(range(len(categories)))
                ax.set_yticklabels(categories)
                ax.set_xlabel("Frecuencia")
                ax.set_ylabel("Categorías")
            else:
                ax.bar(range(len(categories)), frequencies, color=color, alpha=0.7)
                ax.set_xticks(range(len(categories)))
                ax.set_xticklabels(categories, rotation=rotation)
                ax.set_xlabel("Categorías")
                ax.set_ylabel("Frecuencia")

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Mejorar el layout
        plt.tight_layout()

        if show:
            plt.show()

        return fig, ax


class DatasetAnalyzer:
    """
    Analizador sin pandas/NumPy.
    - Usa set_data(dict) para cargar columnas como listas.
    """

    def __init__(self):
        self.tables: Dict[str, FrequencyTable] = {}
        self.data: Dict[str, List[Any]] = {}

    def set_data(self, data: Dict[str, List[Any]]) -> None:
        self.data = {k: list(v) for k, v in data.items()}

    def analyze_column(self, name: str, col: List[Any],
                       bins: Optional[Union[int, List[Number]]] = None) -> Optional[FrequencyTable]:
        vec = list(col)
        if len(vec) == 0:
            return None
        table = FrequencyTable(vec, bins=bins)
        self.tables[name] = table
        return table

    def analyze_all(self, auto_bin: int = 20) -> None:
        for name, vec in self.data.items():
            numeric_vals = [x for x in vec if _is_number(x)]
            uniques = len(set(str(x) for x in vec if x is not None))
            if numeric_vals and uniques > auto_bin:
                bins = max(5, min(10, uniques // 2))
            else:
                bins = None
            self.analyze_column(name, vec, bins=bins)

    def summary_report(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for name, t in self.tables.items():
            s = t.stats
            row: Dict[str, Any] = {
                "Columna": name,
                "Tipo": "Numérico" if "mean" in s else "Categórico",
                "Observaciones": s["observations"],
                "Únicos": s["unique"],
                "Moda": s["mode"],
                "Frecuencia_moda": s["max_freq"],
                "Moda_%": round((s["max_freq"] / max(s["observations"], 1)) * 100, 2),
            }
            if "mean" in s:
                row.update({
                    "Media": round(float(s["mean"]), 4),
                    "Mediana": round(float(s["median"]), 4),
                    "Valor_moda": round(float(s["mode_val"]), 4) if s["mode_val"] is not None else None,
                    "Varianza": round(float(s["variance"]), 4),
                    "DesvEst": round(float(s["std_dev"]), 4),
                    "Rango": round(float(s["range"]), 4),
                    "Q1": round(float(s["q1"]), 4),
                    "Q3": round(float(s["q3"]), 4),
                    "RIQ": round(float(s["iqr"]), 4),
                    "CoefVar": round(float(s["coef_var"]), 4) if s["coef_var"] is not None else None,
                    "Curtosis": round(float(s["kurtosis"]), 4)
                })
            rows.append(row)

        # Imprime como tabla de texto alineada (no listas/dicts)
        if rows:
            base_cols = ["Columna", "Tipo", "Observaciones", "Únicos", "Moda", "Frecuencia_moda", "Moda_%",
                         "Media", "Mediana", "Valor_moda", "Varianza", "DesvEst", "Rango",
                         "Q1", "Q3", "RIQ", "CoefVar", "Curtosis"]
            cols = [c for c in base_cols if any(c in r and r.get(c) is not None for r in rows)]
            widths = {c: len(c) for c in cols}
            for r in rows:
                for c in cols:
                    val = r.get(c, "")
                    sval = "" if val is None else str(val)
                    if len(sval) > widths[c]:
                        widths[c] = len(sval)
            header = " | ".join(c.ljust(widths[c]) for c in cols)
            sep = "-+-".join("-" * widths[c] for c in cols)
            print(header)
            print(sep)
            for r in rows:
                line = " | ".join(
                    ("" if r.get(c) is None else str(r.get(c))).ljust(widths[c]) for c in cols
                )
                print(line)

        return

    def plot_data(self, col: str, kind: str = "bar", **kwargs) -> Dict[str, Any]:
        if col not in self.tables:
            raise ValueError(f"Columna '{col}' no ha sido analizada.")
        return self.tables[col].plot_data(kind=kind, bins=kwargs.get("bins"))

    def plot(self, col: str, kind: str = "bar", **kwargs):
        """
        Grafica una columna ya analizada usando matplotlib.
        """
        if col not in self.tables:
            raise ValueError(f"Columna '{col}' no ha sido analizada.")

        # Si no se proporciona título, usar uno por defecto
        if 'title' not in kwargs:
            kwargs['title'] = f"Distribución {col}"

        return self.tables[col].plot(kind=kind, **kwargs)

    def plot_multiple(self, columns: List[str], kind: str = "bar",
                      figsize: Tuple[int, int] = (15, 10), bins: Optional[int] = None, **kwargs):
        """
        Grafica múltiples columnas en subplots usando matplotlib.
        """
        if not _MATPLOTLIB_AVAILABLE:
            raise RuntimeError("matplotlib no está disponible en el entorno.")

        # Filtrar columnas que existen
        valid_cols = [col for col in columns if col in self.tables]
        if not valid_cols:
            raise ValueError("Ninguna de las columnas especificadas ha sido analizada.")

        n_cols = len(valid_cols)
        n_rows = (n_cols + 1) // 2  # 2 gráficos por fila
        n_subplot_cols = 2 if n_cols > 1 else 1

        fig, axes = plt.subplots(n_rows, n_subplot_cols, figsize=figsize)
        if n_cols == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)

        for i, col in enumerate(valid_cols):
            row = i // 2
            col_idx = i % 2

            if n_cols == 1:
                ax = axes[0]
            else:
                ax = axes[row, col_idx]

            # Obtener datos para graficar
            pdata = self.tables[col].plot_data(kind=kind, bins=bins)

            if pdata.get("kind") == "hist":
                ax.hist(pdata["x"], bins=pdata["bins"], alpha=0.7, edgecolor='black')
                ax.set_xlabel("Valores")
                ax.set_ylabel("Frecuencia")
            else:
                categories = pdata["category"]
                frequencies = pdata["frequency"]

                if len(categories) > 6:
                    ax.barh(range(len(categories)), frequencies, alpha=0.7)
                    ax.set_yticks(range(len(categories)))
                    ax.set_yticklabels(categories)
                    ax.set_xlabel("Frecuencia")
                else:
                    ax.bar(range(len(categories)), frequencies, alpha=0.7)
                    ax.set_xticks(range(len(categories)))
                    ax.set_xticklabels(categories, rotation=45)
                    ax.set_ylabel("Frecuencia")

            ax.set_title(f"Distribución: {col}", fontweight='bold')
            ax.grid(True, alpha=0.3)

        # Ocultar subplots vacíos
        if n_cols % 2 == 1 and n_cols > 1:
            axes[n_rows - 1, 1].set_visible(False)

        plt.tight_layout()
        plt.show()

        return fig, axes
