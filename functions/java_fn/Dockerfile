FROM eclipse-temurin:17-jdk-alpine AS build
WORKDIR /app
COPY Handler.java .
RUN javac Handler.java

FROM eclipse-temurin:17-jre-alpine
WORKDIR /app
COPY --from=build /app/Handler.class .
EXPOSE 8080
CMD ["java", "-Xms32m", "-Xmx128m", "Handler"]
