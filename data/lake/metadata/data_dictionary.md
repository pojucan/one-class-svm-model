# Data Dictionary — Raman Spectroscopy Dataset (Graphene/Carbon Materials)

## 1. Visão geral do dataset

Este dataset contém descritores extraídos de espectros Raman de materiais carbonosos, possivelmente grafeno ou estruturas sp², com bandas características D (~1350 cm⁻¹), G (~1580 cm⁻¹), 2D (~2700–3200 cm⁻¹) e bandas combinadas (D*, D+D', 2D').

Cada linha representa uma amostra experimental identificada por `nome_amostra`.

O conjunto inclui parâmetros espectrais ajustados por decomposição de picos (peak fitting), como posição central, largura (FWHM), amplitude e área integrada.

---

## 2. Estrutura das variáveis

### 2.1 Identificação da amostra

| Variável | Tipo | Descrição |
|----------|------|-----------|
| `nome_amostra` | string | Identificador único da amostra experimental |

---

## 2.2 Banda D (~1350 cm⁻¹) — defeitos estruturais

| Variável | Tipo | Unidade | Descrição |
|----------|------|--------|-----------|
| `D*_centro` | float | cm⁻¹ | Posição do pico D* (modo relacionado a defeitos ou vibrações modificadas) |
| `D*_fwhm` | float | cm⁻¹ | Largura à meia altura da banda D* |
| `D*_amplitude` | float | a.u. | Amplitude do pico D* |
| `D*_area` | float | a.u. | Área integrada do pico D* |
| `D_centro` | float | cm⁻¹ | Posição central da banda D (defeitos estruturais em carbono sp²) |
| `D_fwhm` | float | cm⁻¹ | Largura à meia altura da banda D |
| `D_amplitude` | float | a.u. | Intensidade máxima da banda D |
| `D_area` | float | a.u. | Área integrada da banda D |

---

## 2.3 Banda G (~1580 cm⁻¹) — estiramento sp²

| Variável | Tipo | Unidade | Descrição |
|----------|------|--------|-----------|
| `G*_centro` | float | cm⁻¹ | Posição central da banda G* |
| `G*_fwhm` | float | cm⁻¹ | Largura da banda G* |
| `G*_amplitude` | float | a.u. | Amplitude da banda G* |
| `G*_area` | float | a.u. | Área integrada da banda G* |
| `G_centro` | float | cm⁻¹ | Posição central da banda G |
| `G_fwhm` | float | cm⁻¹ | Largura à meia altura da banda G |
| `G_amplitude` | float | a.u. | Intensidade da banda G |
| `G_area` | float | a.u. | Área integrada da banda G |

---

## 2.4 Banda 2D (~2700–3200 cm⁻¹) — ordem e empilhamento

| Variável | Tipo | Unidade | Descrição |
|----------|------|--------|-----------|
| `2D_centro` | float | cm⁻¹ | Posição central da banda 2D |
| `2D_fwhm` | float | cm⁻¹ | Largura da banda 2D |
| `2D_amplitude` | float | a.u. | Intensidade do pico 2D |
| `2D_area` | float | a.u. | Área integrada da banda 2D |

---

## 2.5 Bandas combinadas / secundárias

| Variável | Tipo | Unidade | Descrição |
|----------|------|--------|-----------|
| `D+D'_centro` | float | cm⁻¹ | Posição da banda combinada D + D' |
| `D+D'_fwhm` | float | cm⁻¹ | Largura da banda D + D' |
| `D+D'_amplitude` | float | a.u. | Intensidade da banda D + D' |
| `D+D'_area` | float | a.u. | Área integrada da banda D + D' |
| `2D'_centro` | float | cm⁻¹ | Posição da banda 2D' |
| `2D'_fwhm` | float | cm⁻¹ | Largura da banda 2D' |
| `2D'_amplitude` | float | a.u. | Intensidade da banda 2D' |
| `2D'_area` | float | a.u. | Área integrada da banda 2D' |

---

## 2.6 Razões e parâmetros derivados

| Variável | Tipo | Unidade | Descrição |
|----------|------|--------|-----------|
| `ID_IG` | float | adimensional | Razão de intensidade entre banda D e G (indicador de defeitos) |
| `I2D_IG` | float | adimensional | Razão entre 2D e G |
| `R_GO` | float | adimensional | Razão relacionada à organização estrutural (G/O ou G-based ratio experimental) |
| `FWHM_G` | float | cm⁻¹ | Largura da banda G (indicador de ordem estrutural) |
| `S_GO` | float | adimensional | Parâmetro espectral derivado associado à ordem sp² |
| `A2D_AG` | float | adimensional | Razão entre áreas 2D e G |

---

## 3. Observações importantes

- Todas as intensidades e áreas estão em **unidades arbitrárias (a.u.)**, dependentes do instrumento.
- As posições de pico estão em **cm⁻¹**.
- O dataset **não contém uma variável alvo (target) explícita**, sendo adequado para:
  - aprendizado não supervisionado (clustering, PCA, embeddings)
  - ou modelagem supervisionada após definição externa de rótulos (ex: tipo de amostra, tratamento, concentração etc.)

---

## 4. Interpretação físico-química

- **Banda D** → defeitos, bordas e desordem estrutural
- **Banda G** → vibração E2g de carbonos sp²
- **Banda 2D** → empilhamento, número de camadas e interação interplanar
- **Razões ID/IG e I2D/IG** → métricas clássicas para grau de defeito e qualidade cristalina

---

## 5. Fonte dos dados

- Técnica: Espectroscopia Raman
- Material: Estruturas carbonosas sp² (ex: grafeno)
- Tipo de dado: parâmetros extraídos por ajuste de picos (peak fitting)