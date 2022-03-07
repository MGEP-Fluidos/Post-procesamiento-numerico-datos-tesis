[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)

[![Made withJupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange?style=for-the-badge&logo=Jupyter)](https://jupyter.org/try)

<h1>Post-procesamiento-datos-tesis</h1>

Los ficheros que se incluyen en este repositorio son para llevar a cabo un primer proyecto de colaboración MGEP/UPM-ETSIAE centrado en analizar los datos obtenidos mediante simulaciones numéricas con un modelo LES sobre un perfil NACA0021 en distintas configuraciones fluidas.

<h2>Propósito del proyecto de colaboración</h2>

El propósito principal del proyecto de colaboración será tratar comparar los datos numéricos obtenidos mediante distintos esquemas numéricos, así como contrastar dichos datos numéricos con los datos experimentales, realizando un análisis temporal de los mismos para ver si se pueden obtener indicadores o patrones que puedan dar cuenta de la naturaleza dinámica de las burbujas de separación laminar (<i>LSB</i>, por sus siglas en inglés).

En principio, los datos experimentales obtenidos en el túnel de viento se han enmarcado dentro de una tesis doctoral, cuya memoria está disponible en <a href=https://www.researchgate.net/publication/354859934_Aerodynamic_Characterization_of_Transitionally-Operating_Airfoils_under_a_set_of_Flow_Conditions_going_from_Ideal_to_Real_Configurations>este enlace</a> (de todas formas, y como se explicita más adelante, este repositorio contiene la documentación mínima relevante para poder entender el marco de la tesis sobre la cual se han realizado las medidas experimentales y las simulaciones). Las configuraciones fluidas contempladas en dicha tesis abarcan cuatro escenarios físicos posibles, a saber:
<ol>
  <li><b>Configuración limpia:</b> es la configuración que se tiene por defecto en el túnel. No se incluyen efectos de turbulencia o rugosidad, y el flujo que entra en la sección de testeo es lo suficientemente uniforme como para despreciar efectos perturbadores. Es la configuración base, en comparación con la cual se van a tratar de ver el efecto que tienen tanto la turbulencia como la rugosidad.</li>
  <li><b>Configuración turbulenta:</b> es la configuración que reproduce efectos turbulentos mediante la inclusión de una malla pasiva a la entrada de la sección de testeo. Se obtienen intensidades de turbulencia de hasta 4%.</li>
  <li><b>Configuración rugosa:</b> es la configuración que reproduce efectos rugosos sobre el perfil mediante la implementación de una banda rugosa en el 10% de la cuerda que se encuentra en el borde de ataque.</li>
  <li><b>Configuración real:</b> es la configuración que reproduce tanto los efectos turbulentos como los rugosos, y el que se supone que es más fiel a lo que ocurre en aplicaciones reales.</li>
</ol>

<h2>Estructura de ficheros y directorios en el repositorio</h2>

La estructura de ficheros y directorios en el repositorio puede subdividirse en tres bloques:
<ol>
  <li>El primer bloque consta de:
    <ol>
      <li>Éste fichero <i>LEEME.md</i>, que es un fichero informativo acerca de qué es y cómo se emplea el repositorio.</li>
      <li>El fichero <i>post-proc.ipynb</i>, que es un fichero de IPython Notebook interactivo. La interactividad se obtiene por medio de <a href=https://mybinder.org/><i>binder</i></a>, un proyecto abierto que tiene por objetivo facilitar la intercambiabilidad de código (sobre todo Python, aunque no exclusivamente) a través de un servidor <i>JupyterHub</i> que aloja el repositorio mismo. Para que el fichero <i>post-proc.ipync</i> pueda ejecutarse correctamente, hace falta especificarle a <i>binder</i> el tipo de dependencias que tiene el código del Notebook; es decir, hay que indicarle qué tipo de módulos de Python ha de instalar a la hora de configurar el servidor de <i>JupyterHub</i>. Estas dependencias se especifican en el fichero <i>requirements.txt</i> que, en principio, no habrá de modificarse.</li>
      <li>El directorio "oculto" <i>.ipynb_checkpoints</i>, que incluye los ficheros de trazabilidad de los cambios que se realizan el el Notebook <i>post-proc.ipynb</i>. Toda vez que se realizan cambios en el Notebook y se guardan, los ficheros en <i>.ipynb_checkpoints</i> se modifican para dar cuenta de ese cambio, y permiten revertir dichos cambios en caso necesario. Es un directorio que, en principio, no habrá de modificarse.</li>
    <li>Los ficheros <i>THESIS_SotA.pdf</i> y <i>THESIS_Clean_config.pdf</i>, que contienen información acerca del estado del arte de la tesis en cuyo marco se han realizado las medidas experimentales y las simulaciones.</li>
    </ol>
  </li>
  <li>El segundo bloque consta de:
    <ol>
      <li>El directorio <i>own_packages</i> que incluye los módulos de Python propios que son necesarios para ejecutar el código del fichero <i>post-proc.ipynb</i>. Dentro del directorio se encuentra:
        <ol>
          <li>El subdirectorio <i>Math_Tools</i>, que incluye el módulo <i>MathTools.py</i>.</li>          
        </ol>
        Hay más información acerca de este módulo en la documentación correspondiente al Notebook, dentro del fichero <i>post-proc.ipynb</i>. De todas formas, este módulo Python no habrá de modificarse.
      </li>
    </ol>
  </li>
  <li>El tercer bloque consta de:
    <ol>
      <li>El directorio raíz, <i>Num_data</i>, que incluye los datos experimentales en crudo obtenidos en el túnel de viento. La estructura de <i>Num_data</i> está especificada en el fichero <i>post-proc.ipynb</i>.</li>
    </ol>
  </li>
</ol>

<h2>Notebook <i>post-proc.ipynb</i> en modo interactivo</h2>

Tal y como se ha mencionado, el proyecto <i>binder</i> permite intercambiar código de forma interactiva. Para poder ver la funcionalidad del código de post-procesado, por tanto, no hace falta tener instalado ninguna distribución de Python ni clonar el repositorio. Es suficiente con clicar en cualquiera de los dos insignias (<i>badges</i>) siguientes.

Esta insignia es para abrir el Notebook <i>post-proc.ipynb</i> en formato interactivo: 

Esta insignia para abrir la instancia raíz del servidor <i>JupyterHub</i>: 

Téngase en cuenta que, debido a dependencias y al peso que tiene el código, las instancias de <i>binder</i> pueden llevar cierto tiempo de configuración y lanzamiento. Una vez que se ha visto la funcionalidad del código y se quieran realizar más cambios, es conveniente clonar el repositorio y trabajar en el equipo local.
