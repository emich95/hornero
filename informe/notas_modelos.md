# Notas para la sección 4 del informe: modelos de ML

Fuente de todos los números: `etapa2/modelos/metricas.json`, generado por `etapa2/entrenar_modelos.py` (18/11/2026).
Gráficos (21/11): `graficos/fig3_attrition.png` (curva ROC de los dos modelos en test e importancia y sentido de
cada variable) y `graficos/fig4_ventas.png` (ventas reales y del modelo en las 40 semanas de test, Aires y Electro
en AMBA). Se regeneran con `python armar_graficos.py` desde `informe/graficos/`.

## Modelo de attrition (empleados.csv)

**Qué predice:** la probabilidad de que un empleado renuncie (`renuncio = Si`, 28,3 % de los 2.000).

**Variables:** área, centro, nivel y modalidad (one-hot) + antigüedad, sueldo vs. mercado, meses sin aumento,
horas extra, distancia, última evaluación, capacitaciones y satisfacción con el clima (estandarizadas).
Dejamos afuera `id_empleado` y `sueldo_bruto`: este último depende del nivel, no mejora el AUC (0,7875 con él,
0,7882 sin él) y la posición del sueldo frente al mercado ya está en `sueldo_vs_mercado_pct`.

**Validación:** train/test estratificado 80/20 y validación cruzada de 5 folds.

| Modelo | AUC CV (5 folds) | AUC test | F1 con umbral 0,5 |
|---|---|---|---|
| Regresión logística (elegido) | 0,788 ± 0,033 | 0,814 | 0,54 |
| LightGBM (descartado) | 0,749 ± 0,038 | 0,787 | 0,52 |

Elegimos la logística porque predice mejor y se interpreta directo: el signo de cada coeficiente dice en qué
sentido actúa la variable. Los efectos en los datos son monótonos (por ejemplo, la tasa de renuncia baja del 46 %
al 14 % entre el quintil con peor y mejor sueldo frente al mercado), que es justo lo que capta bien un modelo lineal.

**Umbral de "en riesgo": 0,5** (probabilidad ≥ 50 %). En test marca al 20,5 % de los empleados, con precisión 0,65
y recall 0,47. La alternativa que maximiza F1 es 0,33 (marca al 34 %, precisión 0,55, recall 0,67).
Usamos 0,5 por ser el más intuitivo; la tool recibe el umbral como parámetro y la respuesta siempre lo explicita.

**Población:** el modelo se entrena con los 2.000 empleados, pero el riesgo se calcula sobre los 1.434 activos
(los que ya renunciaron ya no están en riesgo). Que esas predicciones sean sobre empleados que el modelo vio al
entrenar podría sesgarlas. Lo medimos: con predicciones fuera de muestra (validación cruzada) también salen
122 activos en riesgo, igual que con el modelo final, así que el sesgo no cambia el resultado.

**Qué encontramos al interpretarlo** (importancia = caída del AUC en test al desordenar la variable;
sentido = odds ratio por unidad):

| Variable | Importancia (caída de AUC) | Efecto por unidad sobre las chances de renunciar |
|---|---|---|
| Sueldo vs. mercado (puntos) | 0,060 | −4,5 % por punto |
| Satisfacción con el clima (1 a 5) | 0,055 | −53 % por punto |
| Meses sin aumento | 0,038 | +5,2 % por mes |
| Horas extra por mes | 0,025 | +5,2 % por hora |
| Antigüedad (años) | 0,014 | −8,3 % por año |
| Capacitaciones en el último año | 0,013 | −21 % por capacitación |
| Distancia (km) | 0,005 | +2,3 % por km |
| Área | 0,004 | Atención al cliente es la de mayor riesgo |
| Última evaluación, nivel, modalidad, centro | < 0,003 | casi sin peso una vez que se controla por lo anterior |

El centro y la modalidad pesan poco porque se superponen con el área (Logística y Operaciones son 100 %
presenciales y trabajan en los CD).

**Escenario de la consulta 4** (acercar los sueldos 10 puntos al mercado: +10 a quien está por debajo, con tope
en 0): de 122 activos en riesgo, **52 dejan de estarlo** (43 %).

## Modelo de ventas (ventas.csv)

**Qué predice:** unidades vendidas por semana de cada categoría en cada región.

**Especificación:** regresión lineal sobre log(unidades), con un nivel propio por categoría y región y un efecto
propio de cada categoría para la temperatura, la lluvia y el precio del cobre. En los datos no hay tendencia ni
estacionalidad que el clima no explique (el residuo promedio por mes da prácticamente 0), así que no sumamos
variables de calendario.

**Validación:** split temporal (entrenamos con las primeras 160 semanas, hasta enero de 2026, y testeamos con las
últimas 40).

| Modelo | MAE (unidades) | MAPE | R² |
|---|---|---|---|
| Lineal log con interacciones por categoría (elegido) | 22,8 | 12,2 % | 0,980 |
| Lineal log con todas las interacciones de a pares (descartado) | 23,1 | 12,2 % | 0,980 |
| LightGBM (descartado) | 26,5 | 18,6 % | 0,975 |

Además de predecir mejor, el lineal **extrapola**: el cobre está hoy en 9.470 USD/t y un +20 % lo lleva a unos
11.400, por encima del máximo histórico (11.160). Un modelo de árboles no puede ir más allá de lo que vio y
devolvería casi la misma venta. Entre los dos lineales elegimos el más simple, porque predice igual.

Error por categoría en test (MAPE): Electro 8,5 %, Cocina 9,2 %, Muebles 11,6 %, Calefacción 12,8 %,
Jardín 13,7 %, Aires 17,3 %.

**Qué encontramos al interpretarlo** (efecto sobre las unidades de cada categoría):

| Categoría | +1 °C | +10 mm de lluvia | +1.000 USD/t de cobre |
|---|---|---|---|
| Aires acondicionados | +18,4 % | 0 % | −15,1 % |
| Calefacción | −14,7 % | 0 % | −1,7 % |
| Jardín | +6,1 % | −11,4 % | −0,9 % |
| Electro | 0 % | 0 % | −13,6 % |
| Cocina | 0 % | 0 % | 0 % |
| Muebles | 0 % | 0 % | +1,0 % |

El clima mueve Aires, Calefacción y Jardín; el cobre pega solo en Electro y Aires; Cocina y Muebles dependen solo
de la región (AMBA vende más del doble que la segunda región en todas las categorías).

**Escenario de la consulta 6** (cobre +20 % sobre las últimas 8 semanas): Electro pasa de 3.344 a 2.529
unidades por semana (**−24,4 %**) y Aires de 419 a 304 (**−27,3 %**), con las cantidades corregidas por el factor
de Duan (ver el detalle técnico).

**Detalle técnico:** al volver de log a unidades, la predicción queda en la mediana, un poco por debajo de la
media. Corregimos las cantidades absolutas con el factor de Duan (1,012); los porcentajes de los escenarios no
cambian.

## Limitaciones (para la sección 8)

- Los datos son sintéticos y los efectos son limpios; con datos reales habría más ruido y más variables.
- Las simulaciones muestran asociaciones, no causalidad: que el modelo asocie mejor sueldo con menos renuncias
  no garantiza que subir sueldos las reduzca en esa medida (clase 2).
- El escenario de cobre +20 % extrapola fuera del rango histórico: el resultado depende de que la relación
  siga siendo log-lineal más allá de lo observado.
- Attrition con umbral 0,5: se le escapa la mitad de los que renuncian (recall 0,47).
