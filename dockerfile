

FROM python

# Set the working directory
WORKDIR /app
# Copy the current directory contents into the container at /app

COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

#Copy all project files into the container.


# Make port 8000 available to the world outside this container
EXPOSE 8000


# define command to run the app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

