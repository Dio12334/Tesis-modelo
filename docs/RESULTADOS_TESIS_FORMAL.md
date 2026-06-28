# Detección automática de daños viales mediante RT-DETR sobre el conjunto RDD2022

### Marco teórico, metodología experimental y resultados

> Versión formal y autocontenida del informe de resultados (reescritura de
> [RESULTADOS_TESIS.md](RESULTADOS_TESIS.md)). Está pensada para ser leída por una persona sin
> formación previa en visión por computador: el §2 introduce desde cero los conceptos necesarios
> para interpretar los resultados. Documentos complementarios: arquitectura del sistema
> ([ARQUITECTURA.md](ARQUITECTURA.md)), detalle metodológico de las mejoras
> ([MEJORAS_METRICAS_RT_DETR.md](MEJORAS_METRICAS_RT_DETR.md)) y comparación con el estado del arte
> ([ANALISIS_PERU_Y_SOTA.md](ANALISIS_PERU_Y_SOTA.md)). Fecha: 2026-06-26.

---

## 1. Introducción

El deterioro de la infraestructura vial —baches, grietas longitudinales y transversales,
agrietamiento tipo piel de cocodrilo y otras formas de corrupción del pavimento— constituye un
problema de mantenimiento costoso y, sobre todo, peligroso. Su inspección tradicional es manual:
un técnico recorre las vías y registra los daños, una tarea lenta, subjetiva y difícil de escalar a
la longitud de una red vial nacional. La automatización de esta inspección a partir de imágenes
captadas desde un vehículo es, por tanto, un objetivo de alto valor práctico.

El presente trabajo aborda esa automatización como un problema de **detección de objetos**: dada una
fotografía de la calzada, el sistema debe localizar cada daño mediante un recuadro y asignarle una
categoría. Para ello se entrena y evalúa el modelo **RT-DETR** (un detector basado en la arquitectura
*Transformer*) sobre **RDD2022**, el conjunto de datos público de referencia en este dominio. El
objetivo del estudio no es únicamente alcanzar una métrica elevada, sino **entender qué decisiones de
diseño mueven el desempeño y por qué**, documentando tanto los cambios que funcionaron como los que
no, de modo que las conclusiones sean reproducibles y defendibles.

Este documento se organiza en tres bloques. El §2 ofrece un marco teórico mínimo que define, desde
cero, los conceptos necesarios para leer el resto: qué es la detección de objetos, cómo se mide su
calidad y qué tipo de modelo es RT-DETR. El §3 describe el protocolo experimental (datos, hardware y
método de evaluación). Los §4 a §7 presentan y discuten los resultados: la progresión global del
modelo, el análisis por categoría de daño y por país, y la posición del trabajo frente al estado del
arte. El §8 cierra con las limitaciones y las líneas de trabajo abiertas.

---

## 2. Marco teórico

### 2.1. De la clasificación a la detección de objetos

En visión por computador conviene distinguir dos tareas. La **clasificación de imágenes** responde a
la pregunta «¿qué hay en esta imagen?» con una única etiqueta global (por ejemplo, «esta foto
contiene un bache»). La **detección de objetos** es más exigente: responde «¿qué hay y dónde?»,
devolviendo, para cada objeto presente, un **recuadro delimitador** (en inglés *bounding box*) que lo
encierra y la **categoría** a la que pertenece. Una misma imagen de calzada puede contener varios
daños de clases distintas, y el detector debe localizarlos todos por separado.

Cada predicción de un detector consta, pues, de tres elementos: las coordenadas del recuadro, la
clase predicha y un **valor de confianza** (un número entre 0 y 1 que expresa cuán seguro está el
modelo de esa detección). Esta confianza es central porque permite **filtrar**: si solo se aceptan
las detecciones por encima de un umbral, se obtienen menos detecciones pero más fiables; si se baja
el umbral, se recuperan más daños a costa de admitir más errores. Gran parte del análisis posterior
gira en torno a la elección de ese punto de operación.

El problema de la detección de daños viales es, dentro de su familia, particularmente difícil por
tres razones que conviene tener presentes: los daños son **objetos pequeños y finos** (una grieta
puede ocupar pocos píxeles de ancho), presentan **alta variabilidad dentro de una misma clase** (no
hay dos baches iguales) y sus **etiquetas son ruidosas e inconsistentes** entre los distintos países
que componen el conjunto de datos. A ello se suma una fuerte **dependencia del dominio**: el aspecto
de una calle, la cámara empleada y las condiciones de captura varían tanto de un país a otro que un
modelo entrenado en uno puede degradarse drásticamente en otro.

### 2.2. Cómo se mide la calidad de un detector

Para evaluar un detector hace falta, primero, un criterio que decida cuándo una detección «acierta».
Ese criterio es la **Intersección sobre la Unión** (IoU, *Intersection over Union*): se superpone el
recuadro predicho con el recuadro real (la *verdad de terreno* o *ground truth*) y se calcula el
cociente entre el área que comparten y el área total que cubren entre ambos. El IoU vale 1 cuando los
recuadros coinciden exactamente y 0 cuando no se solapan. Se fija entonces un **umbral de IoU**
(habitualmente 0,5): si una predicción supera ese umbral respecto a un objeto real de la misma clase,
se considera un acierto.

A partir de ese criterio se cuentan tres cantidades: los **verdaderos positivos** (VP, daños reales
correctamente detectados), los **falsos positivos** (FP, detecciones que no corresponden a ningún
daño real) y los **falsos negativos** (FN, daños reales que el modelo no detectó). Con ellas se
definen las dos métricas fundamentales:

- La **precisión** (*precision*) = VP / (VP + FP) responde a «de todo lo que el modelo detectó, ¿qué
  fracción era correcta?». Una precisión alta significa pocas falsas alarmas.
- La **exhaustividad** o **recall** = VP / (VP + FN) responde a «de todos los daños que existían, ¿qué
  fracción encontró el modelo?». Un recall alto significa que se escapan pocos daños.

Ambas están en tensión: bajar el umbral de confianza aumenta el recall pero suele reducir la
precisión, y viceversa. El **F1** resume ese equilibrio como la media armónica de las dos
(F1 = 2·P·R / (P + R)); es alto solo cuando precisión y recall lo son simultáneamente. Como el F1
depende del umbral de confianza elegido, en este trabajo se reporta el **mejor F1** obtenido al
barrer todos los umbrales posibles (*best-F1*), que refleja el potencial del modelo con su punto de
operación óptimo.

La métrica principal de la literatura de detección, sin embargo, es el **AP** (*Average Precision*) y
su promedio entre clases, el **mAP** (*mean Average Precision*). El AP de una clase se obtiene
variando el umbral de confianza de 0 a 1, registrando la pareja (precisión, recall) en cada punto
—lo que traza la llamada **curva precisión-recall**— y midiendo el **área bajo esa curva**. Un AP
cercano a 1 indica que el modelo mantiene alta precisión incluso al exigir alto recall. El mAP es
simplemente la media de los AP de todas las clases, y tiene la virtud de resumir el desempeño en un
único número **independiente del umbral**.

Se emplean dos variantes del mAP que conviene no confundir. El **mAP@0.5** usa un único umbral de IoU
de 0,5 para decidir los aciertos; es una medida relativamente indulgente que premia sobre todo
*encontrar* el daño. El **mAP@0.5:0.95** promedia el resultado sobre diez umbrales de IoU (de 0,50 a
0,95); al exigir solapamientos cada vez más estrictos, penaliza los recuadros mal ajustados y mide,
en la práctica, la **calidad de la localización**. Un modelo puede tener buen mAP@0.5 (detecta los
daños) y a la vez un mAP@0.5:0.95 modesto (los recuadros no encajan con precisión), situación
frecuente en objetos pequeños y de contorno difuso como las grietas.

### 2.3. Familias de detectores: de las redes convolucionales a los *Transformers*

Históricamente, los detectores de objetos se han construido sobre **redes neuronales convolucionales**
(CNN), que extraen patrones visuales locales mediante filtros que recorren la imagen. Dentro de este
paradigma conviven dos enfoques. Los detectores **de dos etapas**, como Faster R-CNN, primero proponen
regiones candidatas y luego las clasifican; son precisos pero relativamente lentos. Los detectores
**de una etapa**, como las familias YOLO y SSD, predicen directamente, sobre una rejilla densa de
posiciones predefinidas (*anclas* o *anchors*), todas las cajas de una sola pasada; son mucho más
rápidos, al precio de un proceso de depuración posterior. Ese proceso es la **supresión de no-máximos**
(NMS, *Non-Maximum Suppression*), un paso heurístico que elimina las detecciones duplicadas que la
rejilla densa produce alrededor de un mismo objeto.

En 2020 apareció un cambio de paradigma con **DETR** (*DEtection TRansformer*), que replantea la
detección como un problema de **predicción de conjuntos**. En lugar de una rejilla de anclas, DETR
emplea la arquitectura *Transformer* —la misma que revolucionó el procesamiento de lenguaje— con un
codificador y un decodificador que operan sobre un conjunto de **consultas de objeto** (*object
queries*) aprendidas. Cada consulta «pregunta» por un posible objeto y produce, como mucho, una
detección. Durante el entrenamiento, un algoritmo de **emparejamiento bipartito** (el método húngaro)
asocia de forma óptima cada predicción con un objeto real, de modo que la red aprende a emitir un
conjunto sin duplicados. La consecuencia más elegante de este diseño es que DETR **no necesita ni
anclas ni NMS**: es un detector «de extremo a extremo». Su inconveniente práctico es que la versión
original es pesada y de convergencia lenta.

**RT-DETR** (*Real-Time DETR*) es la evolución que vuelve operativo ese paradigma. Conserva las
ventajas de DETR —ausencia de anclas y de NMS, formulación de extremo a extremo— pero rediseña sus
componentes para alcanzar velocidad de **tiempo real**. Su pieza central es un **codificador híbrido
eficiente** que separa la interacción de características dentro de una misma escala de la fusión entre
escalas distintas, reduciendo drásticamente el coste de cómputo del codificador, junto con una
**selección de consultas guiada por IoU** que mejora la calidad de las detecciones iniciales. El
resultado es un detector que iguala o supera a las variantes de YOLO comparables en la relación
precisión-velocidad, manteniendo la naturaleza limpia de los *Transformers*. En este trabajo se
emplea la variante **RT-DETR-L** (de *large*, unos 32 millones de parámetros), por ofrecer un buen
equilibrio entre capacidad y coste para el hardware disponible.

### 2.4. Entrenamiento: conceptos que intervienen en las decisiones de diseño

Entrenar un detector consiste en ajustar sus millones de parámetros para que sus predicciones se
acerquen a las etiquetas reales. A continuación se introducen los mecanismos concretos que, más
adelante, serán objeto de decisión y análisis.

**Aprendizaje por transferencia y pesos preentrenados.** Entrenar desde cero exigiría una cantidad de
datos y de cómputo enorme. En su lugar se parte de un modelo **preentrenado en COCO**, un gran
conjunto de detección genérica: su columna vertebral (*backbone*) ya ha aprendido a reconocer bordes,
texturas y formas básicas, y solo resta **adaptarlo** a la tarea específica de daños viales. Esta
práctica, el *transfer learning*, es estándar y se asume en todo el estudio.

**La función de pérdida.** El entrenamiento se guía minimizando una **función de pérdida** que mide
el error de las predicciones. En un detector tiene dos componentes. El término de **clasificación**
penaliza asignar la clase equivocada; suele emplearse la *focal loss*, diseñada para que los
numerosísimos ejemplos «fáciles» de fondo no dominen el aprendizaje, con un parámetro `focal_alpha`
que equilibra el peso de positivos y negativos. Asociado a él, el **peso de no-objeto**
(`no_object_weight`) regula cuánto se penaliza a las consultas que deben quedar «vacías»: un valor
**alto** empuja al modelo a ser conservador (predecir pocas cajas, lo que favorece la precisión) y un
valor **bajo** lo anima a arriesgar más detecciones (lo que favorece el recall). El segundo
componente es el de **regresión del recuadro**, que mide cuánto se desvían las coordenadas predichas
de las reales mediante un error L1 sobre las coordenadas y una pérdida **GIoU** (IoU generalizado) que
alinea las cajas. La ponderación relativa de estos términos es una palanca de diseño.

**Aumento de datos.** Para que el modelo generalice y no se limite a memorizar, se aplican
transformaciones aleatorias a las imágenes de entrenamiento (*data augmentation*): volteo horizontal,
perturbación de color (HSV), cambios de escala y traslación, y técnicas más agresivas como el
**mosaico** (componer una imagen a partir de cuatro, reduciéndolas para que quepan), el **mixup**
(superponer dos imágenes semitransparentes) o el **borrado aleatorio** (*random erasing*: ocultar un
parche rectangular). Estas técnicas regularizan, pero no son neutrales: como se discutirá, el mosaico
**reduce** el tamaño aparente de los objetos, lo que resulta contraproducente cuando los objetos ya
son diminutos.

**Promedio móvil exponencial de pesos (EMA).** En lugar de evaluar los pesos exactos que va dejando el
optimizador —que oscilan de un paso a otro—, se mantiene una **copia suavizada** que es un promedio
exponencial de los pesos a lo largo del entrenamiento (`ema_decay` controla la inercia de ese
promedio). Esa copia, más estable, suele generalizar mejor, y es la que se guarda y evalúa. Es una
práctica estándar de la receta oficial de RT-DETR.

**Tasa de aprendizaje discriminativa.** La **tasa de aprendizaje** (*learning rate*) regula la
magnitud de cada ajuste. Como la columna vertebral ya viene preentrenada y conviene no «estropear» lo
que sabe, se le asigna una tasa **menor** (`backbone_lr`) que a la cabeza de detección
(`head_lr`), que sí debe aprender la tarea nueva desde casi cero. Además, la tasa no es constante: se
emplea un **calentamiento** inicial (*warmup*, subida gradual) seguido de un **decaimiento coseno**
(descenso suave hasta casi cero), lo que permite asentar el entrenamiento al principio y afinarlo con
precisión al final.

**Sobreajuste y parada temprana.** Si se entrena demasiado, el modelo empieza a **memorizar** los
ejemplos de entrenamiento en vez de aprender patrones generales: la pérdida de entrenamiento sigue
bajando, pero la de validación se estanca o empeora. Ese fenómeno es el **sobreajuste**
(*overfitting*), y la distancia creciente entre ambas pérdidas (*gap* train-val) es su señal de
alarma. La **parada temprana** (*early stopping*) detiene el proceso cuando la validación deja de
mejorar y conserva el mejor punto, evitando degradar el modelo.

**Aumento en tiempo de inferencia (TTA).** Por último, en el momento de evaluar se puede ejecutar el
modelo varias veces sobre versiones transformadas de la misma imagen (volteos, distintas escalas) y
**fusionar** las detecciones. Este *Test-Time Augmentation* suele arañar algo de precisión a cambio
de multiplicar el coste de cómputo, por lo que es una técnica de **reporte de resultados** más que de
despliegue en tiempo real.

### 2.5. El conjunto de datos RDD2022

**RDD2022** (*Road Damage Dataset 2022*) es el conjunto público de referencia para esta tarea. Reúne
decenas de miles de imágenes de calzada captadas en varios países —entre ellos Japón, India, Estados
Unidos, Chequia, China y Noruega—, cada una anotada con los recuadros y las categorías de los daños
presentes. Es el conjunto sobre el que se celebran los retos internacionales (CRDDC), lo que lo
convierte en la vara de medir natural del campo.

Su carácter multinacional es, a la vez, su riqueza y su principal dificultad. Las imágenes de
distintos países proceden de **cámaras, resoluciones y criterios de etiquetado heterogéneos**, lo que
introduce la dependencia del dominio ya mencionada y obliga a tomar decisiones explícitas sobre qué
subconjuntos incluir. En este trabajo, las categorías de daño se unifican en **cinco clases** comunes
a todos los países —agrietamiento longitudinal, agrietamiento transversal, agrietamiento tipo piel de
cocodrilo (*alligator*), bache (*pothole*) y otras corrupciones— para que la taxonomía sea coherente.

---

## 3. Metodología experimental

**Entorno de cómputo.** Todos los experimentos se ejecutaron en una estación portátil equipada con una
GPU **NVIDIA RTX 4070 Laptop de 8 GB de memoria de vídeo**, bajo Windows 11, en un entorno Python 3.12
con PyTorch 2.12. La limitación de 8 GB de memoria de vídeo es relevante porque condiciona de forma
directa la resolución de entrada y el tamaño de lote (*batch*) que es posible emplear, y por tanto
varias de las decisiones de diseño descritas más adelante.

**Selección de países.** De los seis países disponibles se entrenó sobre **cuatro** —Japón, India,
Estados Unidos y Chequia—, excluyendo China y Noruega. Esta exclusión no es arbitraria sino
**empírica**: incluir China degradaba el desempeño global porque una parte de sus imágenes son tomas
aéreas con dron, de distribución muy distinta a las captadas desde vehículo; y Noruega, cuyas imágenes
tienen una resolución y una relación de aspecto radicalmente diferentes al resto, también arrastraba
la métrica. La recuperación de Noruega mediante un preprocesamiento adecuado es, precisamente, una de
las líneas de trabajo abiertas (§8).

**Partición de los datos y protocolo de evaluación.** Las etiquetas del conjunto de **test oficial** de
RDD2022 no son públicas (se evalúan en el servidor del reto), de modo que, para poder medir de forma
autónoma, se reservó el **20 % del conjunto de entrenamiento como validación**, con una semilla
aleatoria fija (42) que garantiza que la partición sea idéntica entre experimentos y, por tanto, que
las comparaciones sean justas. Todas las cifras de este informe corresponden a ese conjunto de
validación (5169 imágenes de los cuatro países). El umbral de IoU para contabilizar aciertos es 0,5
salvo cuando se indica explícitamente el mAP@0.5:0.95. Es importante subrayar —y se retoma en el §7—
que este protocolo (cuatro países, validación extraída del entrenamiento) **no es directamente
comparable** con las cifras del reto oficial, que se calculan sobre el test cerrado de seis países.

**Resolución de entrada.** El modelo base operaba a 640 píxeles; a partir de la segunda iteración se
elevó a **768 píxeles**, una decisión cuya motivación se detalla en el §5. A mayor resolución, los
daños finos conservan más detalle, pero también crece el consumo de memoria, lo que obligó a reducir
el tamaño de lote para no exceder los 8 GB disponibles.

**Reproducibilidad.** Cada experimento se define íntegramente por un fichero de configuración YAML, y
las tablas y figuras de este informe se regeneran con `python -m model.tools.consolidate_results` a
partir de los informes de evaluación (`val_evaluation_report.json`) de cada corrida.

---

## 4. Resultados: progresión global del modelo

El desarrollo siguió una secuencia de configuraciones acumulativas, cada una motivada por el
diagnóstico de la anterior. El punto de partida (**v1**) es un RT-DETR-L preentrenado en COCO,
operando a 640 píxeles, que alcanza un mAP@0.5 de 0,591. La segunda configuración (**v2**) introduce
de forma conjunta tres cambios fundamentados en el estado del arte —una función de pérdida reorientada
hacia el recall, el aumento de resolución a 768 píxeles, la tasa de aprendizaje discriminativa y el
promedio EMA— y produce el mayor salto del estudio. Sobre ella, la configuración **R** añade
regularización para contener el sobreajuste, y la evaluación con **TTA** aporta una última mejora
marginal. La configuración **RC**, que ensaya un balanceo por clase, se incluye por completitud pese a
resultar contraproducente.

La siguiente tabla resume el desempeño de cada configuración sobre el conjunto de validación. Desde v2
el modelo entrena y evalúa a 768 píxeles.

| Configuración | Cambios respecto a la anterior | mAP@0.5 | mAP@.5:.95 | Precisión | Recall | F1@0.5 | best-F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| Base (v1) | RT-DETR-L a 640 px | 0,5909 | 0,2964 | 0,6351 | 0,5620 | 0,5963 | 0,5963 |
| v2 | pérdida pro-recall, 768 px, LR discriminativa, EMA | 0,6353 | 0,3246 | 0,5871 | 0,6549 | 0,6191 | 0,6262 |
| R | v2 + *weight decay* 5e-4 + borrado aleatorio | **0,6420** | **0,3281** | 0,5809 | 0,6562 | 0,6163 | 0,6258 |
| RC | R + balanceo por clase | 0,6394 | 0,3242 | 0,5563 | 0,6722 | 0,6088 | 0,6231 |
| **R + TTA** | R + aumento en inferencia | **0,6490** | 0,3271 | 0,5044 | **0,7249** | 0,5949 | **0,6262** |

En conjunto, el recorrido del modelo base al final del estudio supone una mejora del **mAP@0.5 de
0,591 a 0,649** (un incremento de 0,058, equivalente al 9,8 %) y, de forma aún más marcada, del
**recall, de 0,562 a 0,725** (un aumento del 29 %). Esta asimetría no es casual: el cuello de botella
diagnosticado era precisamente la incapacidad del modelo base para *encontrar* los daños, y la línea
de trabajo se orientó deliberadamente a corregirlo. La figura
[progresion_map_recall.png](../results/thesis/progresion_map_recall.png) ilustra esta progresión
conjunta de mAP y recall.

Conviene leer con cuidado el comportamiento del F1. Al introducir el TTA, el F1 calculado a un umbral
de confianza fijo (0,5) **desciende**, lo que podría interpretarse erróneamente como un empeoramiento.
En realidad, el TTA desplaza el punto de operación del modelo hacia el recall, de modo que el umbral
óptimo se mueve; al re-seleccionarlo (en torno a 0,60), el **mejor F1 del barrido se mantiene
equivalente (0,626)**. Este matiz ilustra por qué, en detección, conviene reportar el mejor F1 del
barrido y no una cifra atada a un umbral arbitrario.

---

## 5. Análisis de las decisiones de diseño

Más allá de las cifras agregadas, el valor del estudio reside en entender **qué cambio produjo qué
efecto**. La siguiente descomposición atribuye a cada decisión su contribución al mAP@0.5 e interpreta
su causa.

| Transición | Δ mAP@0.5 | Naturaleza |
|---|---:|---|
| v1 → v2 | **+0,044** | El salto principal: pérdida pro-recall + resolución 768 px |
| v2 → R | +0,007 | Regularización (combate el sobreajuste) |
| R → R+TTA | +0,007 | Aumento en inferencia (solo evaluación) |
| R → RC | **−0,003** | Balanceo por clase: efecto negativo |

**El salto principal (v1 → v2): reorientar la pérdida y subir la resolución.** El diagnóstico del
modelo base reveló que su limitación no era la precisión —que rondaba el 0,59, equilibrada con el
recall— sino el recall: dejaba sin detectar más de la mitad de los baches. La causa estaba en la
configuración de la función de pérdida, que se había ajustado para *reducir falsas alarmas*: un peso
de no-objeto alto (0,3 frente al valor por defecto de 0,1) y un `focal_alpha` bajo. Ambos parámetros,
como se explicó en el §2.4, empujan al modelo a ser conservador y, por tanto, a **sacrificar recall**,
exactamente lo contrario de lo que el problema requería. Devolverlos a una configuración pro-recall
(peso de no-objeto 0,1, `focal_alpha` 0,25) liberó la capacidad de detección del modelo. De forma
complementaria, elevar la resolución de 640 a 768 píxeles atacó la otra cara del mismo problema: los
daños pequeños —baches diminutos, grietas finas— se vuelven reconocibles solo si conservan suficiente
detalle, y a 640 píxeles muchos se perdían en el remuestreo. La conjunción de ambos cambios, junto con
la tasa de aprendizaje discriminativa y el EMA, explica el grueso de la mejora de todo el estudio. Que
el efecto sea **interpretable** —el recall sube y la clase «bache» mejora de forma desproporcionada—
refuerza la confianza en que la causa identificada es la correcta.

**La regularización (v2 → R): una mejora real pero modesta.** Una vez que v2 alcanzó su techo, se
observó que el modelo empezaba a sobreajustar alrededor de la época 16: la pérdida de validación
tocaba fondo y volvía a subir mientras la de entrenamiento seguía cayendo. Para contenerlo se reforzó
la regularización mediante un mayor decaimiento de pesos (*weight decay* de 1e-4 a 5e-4) y el borrado
aleatorio de parches. El efecto fue **positivo pero pequeño** (+0,007 de mAP), lo que admite una
lectura precisa: el sobreajuste existía y era recuperable, pero no era la limitación dominante. La
regularización quedó, en la práctica, casi agotada como palanca de mejora.

**El aumento en inferencia (R → R+TTA): una técnica de reporte.** Aplicar TTA en la evaluación elevó el
mAP@0.5 en otros 0,007 y, sobre todo, el recall en 0,069 (de 0,656 a 0,725), pero **no aumentó el F1
máximo**: como ya se discutió, solo trasladó el punto de operación hacia el recall. Su coste es además
considerable —multiplica por seis el tiempo de inferencia—, lo que entra en conflicto directo con la
razón de ser de RT-DETR, que es operar en tiempo real. La conclusión metodológica es clara: el TTA es
legítimo para **reportar** la mejor cifra posible del modelo, pero el sistema se **despliega sin él**.

**El balanceo por clase (R → RC): un experimento negativo, deliberadamente conservado.** Resultaba
tentador atribuir el bajo desempeño en la clase «bache» a su menor frecuencia en los datos y
corregirlo muestreando esa clase con mayor probabilidad. El experimento RC hizo justamente eso y
**empeoró** el mAP (−0,003), además de degradar el AP del propio bache. La interpretación es
instructiva: el desequilibrio entre clases era leve (un factor de 2,4) y el verdadero cuello de
botella del bache **no era su frecuencia sino su resolución** —su pequeño tamaño en la imagen—, ya
atacado por el aumento a 768 píxeles. Forzar el muestreo con reemplazo solo introdujo un sobreajuste
adicional sin tocar la causa real. Se conserva este resultado negativo porque delimita con precisión
dónde *no* estaba el problema, lo que es tan valioso como saber dónde sí estaba.

Conviene registrar, además, dos decisiones tempranas que conformaron la configuración de partida y que
también se sustentan en evidencia. La primera es la **desactivación del mosaico y el mixup**: aunque
son aumentos habituales y útiles en detección general, en este dominio resultaron **perjudiciales**,
porque el remuestreo que implican reduce aún más unos objetos que ya son diminutos; los datos lo
confirmaron (0,592 sin mosaico frente a 0,550 con él, en igualdad de condiciones). La segunda es la ya
mencionada **exclusión de China y Noruega**, por desplazamiento de dominio y por incompatibilidad de
resolución, respectivamente.

Por último, merece una nota la pregunta —natural— de si bastaba con **entrenar más tiempo**. Se
comprobó empíricamente que no: al reanudar el mejor modelo más allá de su punto óptimo, el mAP se
mantuvo plano, la pérdida de validación empezó a subir y la distancia train-val se disparó, señales
inequívocas de que el modelo había alcanzado su techo de generalización para esa configuración y
estaba **sobreajustando, no subentrenando**. La vía de mejora, por tanto, no era la duración del
entrenamiento sino los cambios de diseño descritos.

---

## 6. Análisis por categoría y por país

### 6.1. Desempeño por categoría de daño

La mejora global no se repartió de forma uniforme entre las cinco clases. La siguiente tabla muestra
el AP@0.5 de cada categoría a lo largo de las configuraciones.

| Clase | Base (v1) | v2 | R | RC | R+TTA |
|---|---:|---:|---:|---:|---:|
| Agrietamiento *alligator* | 0,667 | 0,706 | 0,701 | 0,697 | **0,713** |
| Agrietamiento longitudinal | 0,603 | 0,632 | 0,637 | 0,635 | **0,637** |
| Otras corrupciones | 0,688 | 0,724 | 0,733 | 0,732 | **0,735** |
| **Bache (*pothole*)** | 0,430 | 0,516 | 0,541 | 0,533 | **0,553** |
| Agrietamiento transversal | 0,566 | 0,598 | 0,598 | 0,599 | **0,607** |

El hallazgo más significativo es la evolución de la clase **bache**, la más difícil con diferencia en
el modelo base (AP 0,430). Su AP creció hasta 0,553, una mejora del 29 % que es, además, la mayor en
términos absolutos. Esto confirma de manera directa la hipótesis del §5: el bache es un objeto pequeño
cuya detección dependía críticamente de la resolución y de una pérdida que no penalizara arriesgar
detecciones; ambos remedios actuaron sobre él de forma desproporcionada. El resto de las clases, ya de
por sí más fáciles (las grietas extensas y las corrupciones ocupan más superficie y son más
reconocibles), mejoraron de forma más contenida. La figura
[ap_por_clase.png](../results/thesis/ap_por_clase.png) representa esta evolución.

### 6.2. Desempeño por país y generalización

Evaluar el modelo por separado en cada país revela su capacidad de **generalización** y expone la
principal limitación que queda abierta. La tabla recoge el mAP@0.5 por país junto al número de
imágenes de entrenamiento disponibles de cada uno.

| País | Imágenes (train) | Base (v1) | R | R+TTA |
|---|---:|---:|---:|---:|
| Japón | 10 506 | 0,604 | 0,648 | **0,657** |
| Estados Unidos | 4 805 | 0,482 | 0,508 | **0,514** |
| India | 7 706 | 0,347 | 0,471 | **0,481** |
| Chequia | 2 829 | 0,266 | 0,300 | **0,328** |

A primera vista, el desempeño parece seguir la cantidad de datos: Japón, el país mejor representado,
es también donde el modelo rinde mejor, y Chequia, el de menor representación, donde peor. Chequia es
un caso claro de **escasez de datos** (*data-starved*): con apenas 2829 imágenes, el modelo no dispone
de suficiente variedad para generalizar, y aquí sí cabría esperar que más datos o un muestreo
compensatorio ayudasen.

Sin embargo, **India rompe esa lectura simple** y constituye el hallazgo más interesante de esta
sección. Con 7706 imágenes —casi el triple que Chequia— su mAP es, no obstante, muy bajo (0,347 en el
modelo base). La cantidad de datos, por tanto, no explica su dificultad: la causa debe ser
**intrínseca al dominio indio o a la calidad de sus etiquetas** (mayor variabilidad de escenas, ruido
o inconsistencia en el anotado). Esta observación tiene una consecuencia metodológica importante:
**balancear los países a ciegas sería arriesgado**, porque dar más peso a India no atacaría su
verdadero problema, y reducir Japón —el país que sostiene la métrica— sería contraproducente. La
brecha entre Japón y Chequia (e India) sigue siendo la principal limitación de generalización del
sistema, y su diagnóstico fino —en particular, una auditoría de las etiquetas de India— queda como
trabajo pendiente.

### 6.3. La dependencia del dominio, ilustrada

Un experimento complementario, descrito en detalle en
[ANALISIS_PERU_Y_SOTA.md](ANALISIS_PERU_Y_SOTA.md), aporta evidencia contundente sobre hasta qué punto
esta tarea depende del dominio. Al evaluar el mejor modelo RDD (mAP 0,642) sobre un conjunto propio de
calles de Perú, su desempeño **se desplomó a 0,11**: un modelo fuerte en un dominio resulta casi
inservible en otro distinto. Afinarlo con datos peruanos casi **duplicó** el desempeño local (hasta
0,21), confirmando que los datos del nuevo dominio aportan, pero ese afinamiento se produjo a costa de
**olvidar** parcialmente RDD (que cayó a 0,477). La lección, que refuerza la interpretación de las
diferencias entre países, es que la detección de daños viales es fuertemente dependiente del dominio y
que un modelo robusto en varios contextos exige entrenamiento conjunto, no un simple afinamiento
sucesivo.

---

## 7. Comparación con el estado del arte

Situar estos resultados frente a la literatura exige, ante todo, **cautela metodológica**, pues las
cifras solo son comparables si se miden sobre el mismo terreno. La referencia «alta» del campo es el
reto **CRDDC'2022**, cuya solución ganadora alcanzó un **F1 de 0,769**. Esa cifra, sin embargo, se
obtuvo sobre el **test oficial de seis países** y, sobre todo, mediante un **conjunto (*ensemble*) de
varios modelos** combinado con aumento en inferencia, fusión ponderada de cajas (WBF) y
**pseudo-etiquetado** del conjunto de test sin anotar. Las soluciones segunda y tercera (un YOLOv5x
multiescala y un YOLOv7 con atención de coordenadas, con F1 de 0,743 y 0,741) comparten esa naturaleza
de maquinaria pesada. No son, por tanto, la referencia adecuada para un **modelo único** entrenado en
una estación portátil.

La comparación pertinente es con la literatura de **modelos transformer individuales** (RT-DETR y sus
derivados) aplicados a daño vial. En esa banda, los trabajos publicados reportan típicamente un
mAP@0.5 en el rango **0,62–0,70** y un F1 en torno a **0,70**. Resultado de este trabajo —un RT-DETR-L
único que alcanza **mAP@0.5 0,649** con TTA— se sitúa de lleno dentro de ese rango, ligeramente por
debajo en F1 (0,626) pero con la salvedad de que el protocolo de medición difiere entre estudios. La
conclusión es que, **como detector transformer individual, el modelo es competitivo con el estado del
arte de su categoría**.

¿Por qué, entonces, no se alcanza el 0,769 del reto? La respuesta no es una deficiencia del modelo,
sino la conjunción de tres factores. El primero, y principal, es que **la comparación no es válida**:
el reto mide F1 sobre el test cerrado de seis países, mientras que este trabajo mide sobre una
validación extraída del entrenamiento de cuatro países, habiendo excluido además los dos más difíciles
(China y Noruega); son tareas distintas. El segundo es que el reto enfrenta **un modelo único contra
una maquinaria de conjunto** con pseudo-etiquetado, que es justamente la palanca que aporta los
últimos puntos y que aquí no se ha empleado. El tercero es que **la tarea es intrínsecamente difícil y
dependiente del dominio** —como el experimento de Perú demuestra de forma drástica—, hasta el punto de
que incluso el mejor resultado del reto (0,77) resulta modesto comparado con los estándares de
*benchmarks* genéricos como COCO. En síntesis, la distancia al estado del arte del reto es de
**método y de protocolo de evaluación, no de calidad del modelo base**.

---

## 8. Limitaciones y líneas de trabajo

El estudio deja varias limitaciones reconocidas y, a partir de ellas, un conjunto de líneas de trabajo
con mayor techo potencial que las ya exploradas.

La limitación metodológica más relevante es que la evaluación se realiza sobre una **validación
extraída del entrenamiento de cuatro países**, lo que impide la comparación directa con las cifras del
reto oficial; reportar sobre el test cerrado de seis países sería el paso natural para una comparación
rigurosa. En cuanto a la generalización, persiste la **brecha entre países**, con India como caso
paradigmático de bajo desempeño no atribuible a la cantidad de datos, cuyo origen —presumiblemente la
calidad del etiquetado— merece una auditoría específica.

Una segunda limitación, más sutil pero importante para interpretar las cifras absolutas, deriva de la
naturaleza del conjunto: RDD2022 se construye a partir de **secuencias de vídeo**, de modo que muchas
imágenes son **fotogramas casi idénticos** tomados con fracciones de segundo de diferencia. Al
particionar de forma aleatoria, es probable que pares de fotogramas casi duplicados queden **repartidos
entre entrenamiento y validación**, una forma de **fuga de información** (*data leakage*) que infla de
manera optimista el desempeño medido. En consecuencia, los valores absolutos reportados —por ejemplo,
el mAP@0.5 de 0,649— deben leerse como un **límite superior**: un protocolo más estricto, que deduplique
por *hash* perceptual o por distancia entre fotogramas antes de la partición, arrojaría con probabilidad
cifras algo menores. Conviene subrayar, no obstante, que esta fuga **no compromete las conclusiones
comparativas** del estudio: como todas las configuraciones comparten exactamente la misma partición, el
efecto de la fuga es común a todas y las diferencias *entre* ellas —la atribución de cada decisión de
diseño— siguen siendo válidas. Es el nivel absoluto, no el orden relativo, lo que queda matizado.

Entre las líneas abiertas, la de mayor recorrido es el **pseudo-etiquetado** del conjunto de test sin
anotar, que fue la técnica decisiva de los ganadores del reto y que no se ha explorado. En la misma
dirección, la construcción de un **conjunto de modelos** (por ejemplo, combinando RT-DETR con una
arquitectura de la familia YOLO mediante fusión ponderada de cajas) suele aportar varios puntos
adicionales. Para el despliegue multinacional —y a la luz del experimento de Perú— el camino correcto
es el **entrenamiento conjunto** de los dominios en lugar del afinamiento sucesivo, que provoca olvido.

Finalmente, se exploró una línea orientada a **recuperar los países hasta ahora excluidos por
incompatibilidad de resolución**, en particular Noruega, cuyas imágenes son mucho más grandes y de
relación de aspecto panorámica (aproximadamente 2:1) frente a las imágenes cuadradas del resto. La
hipótesis de partida era que su bajo desempeño se debía sobre todo a un **artefacto del
preprocesamiento** —la deformación 2:1 que introduce el redimensionado cuadrado— y que corregirla con
un preprocesamiento por **letterbox** (que preserva la relación de aspecto añadiendo relleno) bastaría
para recuperarla. Para comprobarlo se ejecutó una batería de tres experimentos controlados a 768
píxeles, sobre la misma receta, comparando el desempeño sin Noruega, con Noruega bajo el redimensionado
cuadrado y con Noruega bajo letterbox.

El resultado **refutó la hipótesis** y, al hacerlo, refinó el diagnóstico. Incluir Noruega degradó el
mAP global con cualquiera de los dos preprocesamientos (0,632 sin Noruega → 0,603 con redimensionado
cuadrado → 0,590 con letterbox), lo que valida la exclusión original. Y, en contra de lo esperado, el
letterbox **empeoró** el desempeño en Noruega (de 0,297 a 0,232) en lugar de mejorarlo. La explicación
está en el presupuesto de píxeles: a una resolución fija de 768, preservar la relación de aspecto 2:1
obliga a encoger la imagen a 768×384 y rellenar el resto con padding, de modo que **se desperdicia la
mitad del lienzo y se reduce a la mitad la resolución vertical** del contenido útil. Como la detección
de daños finos depende críticamente de esa resolución, la corrección de la distorsión no compensa la
pérdida de detalle. El cuello de botella de Noruega es, por tanto, la **resolución (el submuestreo de
sus imágenes de 8 megapíxeles), no la distorsión de aspecto**. La consecuencia metodológica —valiosa
más allá de este caso— es que «arreglar» la relación de aspecto mediante letterbox solo rinde si se
acompaña de **mayor resolución de entrada**, que es justamente lo que no permite el límite de 8 GB de
memoria de la estación portátil. La recuperación de Noruega queda, así, condicionada a entrenar a
resolución más alta (1024 píxeles o superior, en una máquina con más memoria) o a recurrir al
**teselado** (*tiling*/SAHI), el procedimiento estándar para imágenes de muy alta resolución; ambos
caminos quedan planteados como trabajo futuro y su detalle se documenta en
[EXPERIMENTO_NORUEGA.md](EXPERIMENTO_NORUEGA.md).

### 8.1. Mejoras del conjunto de datos como trabajo futuro (priorizadas por palanca)

Las líneas anteriores se centran en el método —pseudo-etiquetado, conjuntos de modelos, mayor
resolución—. Existe, en paralelo, un conjunto de **intervenciones sobre los datos** cuyo orden de
prioridad se desprende directamente del diagnóstico de este trabajo: el cuello de botella es
**encontrar el daño pequeño y la calidad de las etiquetas**, no la confusión entre clases. Un principio
gobierna todas ellas y conviene hacerlo explícito: cualquier cambio en los datos debe aplicarse **de
forma idéntica a todas las configuraciones comparadas**, pues de lo contrario se rompe la validez de las
comparaciones que sostienen las conclusiones.

La intervención de mayor palanca es la **mejora de la calidad y la consistencia de las etiquetas**. En
primer lugar, la **deduplicación de fotogramas casi idénticos** antes de la partición —ya señalada como
limitación—, que entrega una cifra honesta y un punto de partida limpio. En segundo lugar, y de forma
destacada, una **auditoría de las etiquetas de India**: como se mostró en el §6.2, India rinde por
debajo de lo que su volumen de datos haría esperar —no es un problema de cantidad, sino con toda
probabilidad de ruido o inconsistencia en el anotado—, de modo que muestrear una porción, cuantificar la
tasa de cajas mal etiquetadas o ausentes y re-anotar (o ponderar a la baja) es el paso individual con
mayor probabilidad de mover la métrica. A ello se suman dos depuraciones: **resolver la clase «otras
corrupciones»**, un cajón de sastre heterogéneo que conviene subdividir en subclases más limpias o, en
su defecto, eliminar para alinearse con la literatura de cuatro clases —con la salvedad de que prescindir
de ella reduciría el mAP en una cantidad estimada de un par de puntos, no medida en este estudio—; y el
**descarte de casos vacíos o ilegibles**, es decir, imágenes que quedan sin cajas válidas tras el
filtrado y cajas por debajo de un tamaño mínimo legible.

La segunda familia de intervenciones consiste en **añadir datos que ataquen directamente el déficit de
recall**. La más alineada con el diagnóstico es la **incorporación de datos dirigidos de la clase
bache** —la más rara y difícil, y aquella sobre la que el balanceo por clase fracasó—: técnicas de
*copy-paste* de baches, o la fusión de una fuente rica en este daño, atacan la causa real (recall y
resolución) allí donde el muestreo no pudo. En segundo término, la **recuperación correcta de los países
excluidos**: para Noruega, y como confirmó el experimento de alta resolución, no basta con corregir la
relación de aspecto, sino que hace falta mayor resolución de entrada (1024 píxeles o más) o teselado;
para China, bastaría con incluir únicamente el subconjunto captado **desde vehículo**, excluyendo las
tomas con dron que motivaron su exclusión. Por último, la **ampliación con conjuntos afines** —N-RDD2024,
una extensión del propio RDD2022, y RDD2020— aumentaría el volumen y la diversidad y mejoraría la
comparabilidad con la literatura, si bien supone un salto de alcance mayor que las depuraciones
anteriores.

En conjunto, estas medidas **elevarían el desempeño absoluto** —en particular el recall sobre baches y
objetos pequeños— y acercarían las cifras al rango del 64–66 % que reporta la literatura comparable;
pero, aplicadas de manera uniforme, **no alterarían las conclusiones comparativas** del estudio. Es, en
suma, una hoja de ruta de datos ordenada por palanca y anclada al diagnóstico de recall y de calidad de
etiquetas establecido en las secciones anteriores, no un catálogo de buenas intenciones.

---

## 9. Configuración y artefactos de referencia

El modelo final corresponde a la configuración **R** (`model/configs/train_rt_detr_v2_R.yaml`):
RT-DETR-L a 768 píxeles, optimizador AdamW, tasa de aprendizaje discriminativa, promedio EMA y la
regularización descrita. Su mejor punto (pesos EMA de la época 15) se conserva en
`checkpoints/rt_detr_v2_R/b5da089a-ab0e-4f69-b72c-10f28487e056/best_model.pt`, y la evaluación con TTA
se reproduce añadiendo las opciones `--tta --tta-scales 0.8333 1.0 1.1667`. Las tablas en formato CSV
y las figuras en PNG residen en `results/thesis/`, y la totalidad de tablas y gráficos de este informe
se regenera con `python -m model.tools.consolidate_results`.
