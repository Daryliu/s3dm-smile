package com.smart3dmap.config;

import com.smart3dmap.driver.DataSourceRegister;
import com.smart3dmap.driver.DynamicDataSource;
import com.smart3dmap.server.config.Config;
import com.zaxxer.hikari.HikariDataSource;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import javax.sql.DataSource;
import java.sql.Connection;
import java.util.HashMap;
import java.util.Map;

@Configuration
public class DbSourceConfig {

    @Bean
    public DataSource getDataSource(final Config config) throws Exception {
        Map options = config.getServiceOptions();
        // 默认数据源配置
        Map dbOptionBase = (Map) options.get("base");
        String dbHostBase = dbOptionBase.get("host").toString();
        String dbPortBase = dbOptionBase.get("port").toString();
        String dbNameBase = dbOptionBase.get("database_name").toString();
        String dbUserBase = dbOptionBase.get("user").toString();
        String dbPasswordBase = dbOptionBase.get("password").toString();
        HikariDataSource dataSourceBase = new HikariDataSource();
        dataSourceBase.setDriverClassName("org.postgresql.Driver");
        dataSourceBase.setJdbcUrl(String.format("jdbc:postgresql://%s:%d/%s?stringtype=unspecified", dbHostBase, Integer.parseInt(dbPortBase), dbNameBase));
        dataSourceBase.setUsername(dbUserBase);
        dataSourceBase.setPassword(dbPasswordBase);
        DataSourceRegister.setProperties(dataSourceBase);

        // 尝试连接数据库
        Connection connection = dataSourceBase.getConnection();
        connection.close();

        Map<String, DataSource> dataSource = new HashMap<>();
        dataSource.put("base", dataSourceBase);
        DynamicDataSource.init(dataSource, "base");
        return DynamicDataSource.getInstance();
    }
}