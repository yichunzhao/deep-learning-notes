


* Gradio is an open-source Python library for creating customizable web-based user interfaces, particularly for machine learning models and computational tools.
* Gradio allows you to create interfaces for machine-learning models with just a few lines of code. It supports various inputs and outputs, such as text, images, ﬁles, and more.
* Gradio interfaces can be shared with others through unique URLs, facilitating easy collaboration and feedback collection.
* Setting up a Gradio interface comprises four steps: writing Python code, creating an interface, launching the web server, and accessing the web interface.
* The key features of Gradio include gr.Textbox for text input/output, gr.Number for numeric inputs, and gr.File for file uploads, enabling multiple file selections.
* Once deployed, users can interact with the interface in real time via a web link.


Gradio is an open-source Python package that allows for the quick building of an application interface. No JavaScript, CSS, or web hosting experience is required! Use the launch() method to launch the application. This method starts a simple web server that serves the interface.

| Term          | Definition |
| ------------- | ------------- |
| Fn            | Fn is the function to wrap. Each function parameter represents an input component, and the function's output should be either a single value or a tuple. Each element within the tuple corresponds directly to a specific output component.  |
| inputs        | The Gradio component(s) to use for the inputs of the function. The number of inputs should match the number of arguments in the function.  |
| outputs        | Outputs are the Gradio component(s) to use for the outputs of the function.  |

