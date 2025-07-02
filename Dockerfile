FROM continuumio/miniconda3:4.10.3p1

# Set the working directory to /app
WORKDIR /main

# Copy the current directory contents into the container at /app
COPY . /main

RUN conda install \
    xarray \ 
    netCDF4 \ 
    bottleneck \
    numpy \
    pandas \
    matplotlib \
    jupyterlab

COPY requirements.txt /tmp/
RUN pip install --requirement /tmp/requirements.txt

# Install any needed packages specified in requirements.txt
#RUN pip install --trusted-host pypi.python.org


# Make port 80 available to the world outside this container
EXPOSE 8080

# Define environment variable
ENV PYTHONPATH "${PYTHONPATH}:/main/lib"


# Run app.py when the container launches
CMD ["jupyter-lab","--ip=0.0.0.0","--no-browser","--allow-root"]
